from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import ManifestDataset, TableTransform, collate, progression_label, read_manifest
from .models.core import build_model


@dataclass
class TrainingConfig:
    seed: int = 3407
    epochs: int = 120
    min_epochs: int = 30
    patience: int = 25
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    min_learning_rate: float = 1e-6
    cluster_weight: float = 0.1
    separation_weight: float = 0.1
    diversity_weight: float = 0.01
    ordinal_weight: float = 0.1


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def task_loss(model, output, batch, task, config):
    if task == 'brain_age':
        return F.mse_loss(output['age'], batch['target'])
    loss = F.cross_entropy(output['logits'], batch['target'])
    for modality, out in output.get('prototypes', {}).items():
        bank = model.local if task == 'progression' else model.banks[modality]
        regularizers = bank.regularization(out, batch['target'])
        loss = loss + sum(getattr(config, key+'_weight')*value for key, value in regularizers.items())
    return loss


def train(manifest, spec, output, *, config=None, device='cpu'):
    config = config or TrainingConfig()
    if config.seed not in (3407, 3408, 3409):
        raise ValueError('Paper seeds are 3407, 3408 and 3409')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = read_manifest(manifest)
    task = spec['task']
    reference_rows = []
    if task == 'brain_age':
        rows = [r for r in rows if r.get('clinical_label') == 'CN']
    elif task == 'progression':
        reference_rows = [r for r in rows if r['split'] == 'TRAIN' and r.get('clinical_label') in ('CN', 'AD')]
        if {r['clinical_label'] for r in reference_rows} != {'CN', 'AD'}:
            raise ValueError('Progression ordinal learning needs CN and AD TRAIN references')
        horizon = spec['kwargs']['horizon_years']
        rows = [r for r in rows if progression_label(r.get('baseline'), r.get('conversion_years'), r.get('followup_years'), horizon) is not None]
    train_rows = [r for r in rows if r['split'] == 'TRAIN']
    val_rows = [r for r in rows if r['split'] == 'VALIDATION']
    if not train_rows or not val_rows:
        raise ValueError('TRAIN and VALIDATION are required')
    transform = TableTransform.fit(train_rows)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    model = build_model(spec).to(device)
    loader = DataLoader(ManifestDataset(train_rows, spec, transform), batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ManifestDataset(val_rows, spec, transform), batch_size=config.batch_size, collate_fn=collate)
    references = None
    if reference_rows:

        reference_rows = [dict(r, label=0 if r['clinical_label'] == 'CN' else 2) for r in reference_rows]
        ref_spec = dict(task='classification', kwargs=dict(route='MRI'))
        references = DataLoader(ManifestDataset(reference_rows, ref_spec, transform), batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, eta_min=config.min_learning_rate)
    history, best, stale = [], float('inf'), 0
    use_bf16 = str(device).startswith('cuda') and torch.cuda.is_bf16_supported()
    for epoch in range(config.epochs):
        model.train(); total = 0.; count = 0
        ref_iterator = iter(references) if references is not None else None
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu', dtype=torch.bfloat16, enabled=use_bf16):
                loss = task_loss(model, model(batch), batch, task, config)
                if references is not None:
                    try: reference = next(ref_iterator)
                    except StopIteration:
                        ref_iterator = iter(references); reference = next(ref_iterator)
                    reference = move(reference, device)
                    loss = loss+config.ordinal_weight*model.ordinal_loss(reference['mri'], reference['target'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward(); optimizer.step()
            n = len(batch['target']); total += float(loss.detach())*n; count += n
        model.eval(); validation, nval = 0., 0
        with torch.no_grad():
            for batch in val_loader:
                batch = move(batch, device); result = model(batch); n = len(batch['target'])
                objective = F.mse_loss(result['age'], batch['target']) if task == 'brain_age' else F.cross_entropy(result['logits'], batch['target'])
                validation += float(objective)*n; nval += n
        score = validation/nval
        history.append(dict(epoch=epoch+1, train_loss=total/count, validation_loss=score))
        if score < best:
            best, stale = score, 0
            torch.save(dict(state_dict=model.state_dict(), spec=spec, table_transform=dict(means=transform.means.tolist(), scales=transform.scales.tolist()),
                            training=asdict(config), selected_epoch=epoch+1, prototype_projection='pending' if task != 'brain_age' else 'not_applicable'), out/'best.pt')
        else: stale += 1
        scheduler.step()
        if epoch+1 >= config.min_epochs and stale >= config.patience:
            break
    (out/'history.json').write_text(json.dumps(history, indent=2))
    (out/'config.json').write_text(json.dumps(dict(model=spec, training=asdict(config), validation_selection='minimum_validation_loss',
                                                calibration='not_fitted', bf16=use_bf16), indent=2))
    return history
