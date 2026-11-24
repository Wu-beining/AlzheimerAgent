from pathlib import Path
import json
import torch
from .data import ManifestDataset, TableTransform, collate, read_manifest, progression_label
from .models.core import build_model
from .preprocessing import file_hash


def project_checkpoint(checkpoint, manifest, output, *, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    spec = saved['spec']; task = spec['task']
    if task == 'brain_age' or spec.get('kwargs', {}).get('route') == 'TABLE':
        raise ValueError('This model has no ROI prototypes')
    rows = [r for r in read_manifest(manifest) if r['split'] == 'TRAIN']
    if task == 'progression':
        rows = [r for r in rows if progression_label(r.get('baseline'), r.get('conversion_years'), r.get('followup_years'), spec['kwargs']['horizon_years']) is not None]
    model = build_model(spec).to(device); model.load_state_dict(saved['state_dict']); model.eval()
    transform = TableTransform(**saved['table_transform'])
    banks = {'mri': model.local} if task == 'progression' else model.banks
    records = []
    for modality, bank in banks.items():
        def batches():
            for row in rows:
                batch = {k: v.to(device) for k, v in collate([ManifestDataset([row], spec, transform)[0]]).items()}
                with torch.no_grad():
                    _, grid = (model.encoder if task == 'progression' else model.encoders[modality])(batch[modality])
                info = row['inputs'][modality]
                metadata = dict(sample_id=f"{row['cohort']}:{row['patient']}:{row['visit']}", split='TRAIN',
                    input_sha256=file_hash(info['path']), spatial=dict(space=info['space'], atlas_version=info['atlas_version']),
                    receptive_field=dict(input_shape=list(batch[modality].shape[-3:]), feature_shape=list(grid.shape[-3:]),
                                         stride_voxels=16, receptive_field_voxels=31, center_offset_voxels=0))
                yield grid, batch[f'{modality}_atlas'], batch['target'], [metadata]
        version = rows[0]['inputs'][modality]['atlas_version'] if rows else None
        if not rows or any(r['inputs'][modality]['atlas_version'] != version for r in rows):
            raise ValueError('Projection requires one consistent TRAIN atlas version')
        records.extend(bank.project(batches(), split='TRAIN', modality=modality, atlas_version=version))
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    saved.update(state_dict=model.state_dict(), prototype_projection='TRAIN_same_class_ROI', projection_records=records)
    torch.save(saved, out/'projected.pt')
    (out/'projection.json').write_text(json.dumps(records, indent=2))
    return str(out/'projected.pt')
