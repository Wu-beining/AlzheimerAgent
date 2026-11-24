from pathlib import Path
import json
import torch
from .calibration import TemperatureCalibration
from .data import ManifestDataset, TableTransform, collate
from .models.core import build_model
from .preprocessing import file_hash


class LocalPredictor:
    def __init__(self, checkpoint, expected_sha256, calibration=None, device='cpu'):
        if file_hash(checkpoint) != expected_sha256:
            raise ValueError('Checkpoint checksum mismatch')
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        self.spec, self.device = saved['spec'], device
        self.model = build_model(self.spec).to(device)
        self.model.load_state_dict(saved['state_dict'], strict=True)
        self.model.eval()
        self.transform = TableTransform(**saved['table_transform'])
        self.calibration = TemperatureCalibration(**calibration) if calibration else None
        self.sha256 = expected_sha256

    @torch.no_grad()
    def predict(self, row):
        batch = collate([ManifestDataset([row], self.spec, self.transform, labeled=False)[0]])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        result = self.model(batch)
        if self.spec['task'] == 'brain_age':
            return dict(predicted_age=float(result['age'][0]), checkpoint_sha256=self.sha256)
        if self.calibration is None:
            raise ValueError('Probability inference requires a frozen calibration artifact')
        probabilities = self.calibration.apply(result['logits'].float().cpu().numpy())[0]
        explanations = {m: dict(contributions=r['contributions'][0].cpu().tolist(),
                                feature_indices=r['locations'][0].cpu().tolist())
                        for m, r in result.get('prototypes', {}).items()}
        return dict(probabilities=probabilities.tolist(), prototypes=explanations,
                    calibration_version=self.calibration.version, checkpoint_sha256=self.sha256)


def classification_members(registry, row, device='cpu'):
    from .memory import LABELS
    results = {}
    for model in registry:
        route = model['route']
        try:
            predictor = LocalPredictor(model['checkpoint'], model['sha256'], model['calibration'], device)
            if predictor.spec['task'] != 'classification' or predictor.spec.get('kwargs', {}).get('route', 'MRI') != route:
                raise ValueError('Model/route mismatch')
            prediction = predictor.predict(row)
            member = dict(model_id=model['id'], status='available', calibration_version=prediction['calibration_version'],
                          probabilities=dict(zip(LABELS, prediction['probabilities'])), explanations=prediction['prototypes'])
        except Exception as exc:
            member = dict(model_id=model['id'], status='failed', error=type(exc).__name__)
        results.setdefault(route, []).append(member)
    return results


def predict_tasks(registry, row, bank, *, task_inputs=None, device='cpu'):

    from .memory import Evidence, LABELS
    from .calibration import BAGCalibration
    task_inputs = task_inputs or {}
    pathways, statuses = {}, []
    seen = set()
    for entry in registry:
        if entry['id'] in seen:
            raise ValueError('Duplicate registry model ID')
        seen.add(entry['id'])
        task = entry.get('task', 'classification')
        try:
            inputs = task_inputs.get(entry['id'], row)
            if inputs.get('patient') != bank.patient or inputs.get('visit') != bank.visit:
                raise ValueError('Model input patient/visit mismatch')
            predictor = LocalPredictor(entry['checkpoint'], entry['sha256'], entry.get('calibration'), device)
            if predictor.spec['task'] != task:
                raise ValueError('Registry task differs from checkpoint')
            prediction = predictor.predict(inputs)
            if task == 'classification':
                route = entry['route']
                if predictor.spec.get('kwargs', {}).get('route', 'MRI') != route:
                    raise ValueError('Registry route differs from checkpoint')
                pathways.setdefault(route, []).append(dict(model_id=entry['id'], status='available',
                    calibration_version=prediction['calibration_version'], probabilities=dict(zip(LABELS, prediction['probabilities'])),
                    explanations=prediction['prototypes']))
            else:
                if task == 'brain_age':
                    content = BAGCalibration(**entry['bag_calibration']).apply(prediction['predicted_age'], inputs['age'])
                    unit, domain = 'years', 'mri'
                elif task == 'progression':
                    content = dict(value=prediction['probabilities'][1], horizon_years=predictor.spec['kwargs']['horizon_years'],
                                   baseline_condition='MCI', calibration_version=prediction['calibration_version'],
                                   prototypes=prediction['prototypes'])
                    unit, domain = 'probability', 'model'
                else:
                    raise ValueError('Unknown registered task')
                hashes = {'checkpoint': entry['sha256']}
                for modality, metadata in inputs.get('inputs', {}).items():
                    hashes[modality] = file_hash(metadata['path'])
                bank.add(Evidence(id=entry['id'], patient=bank.patient, visit=bank.visit, task=task,
                    source=dict(version=entry.get('version', entry['id']), domain=domain, input_hashes=hashes),
                    content=content, unit=unit, time=bank.decision_time, status='verified'))
            statuses.append(dict(model_id=entry['id'], task=task, status='available'))
        except Exception as exc:
            statuses.append(dict(model_id=entry['id'], task=task, status='failed', error=type(exc).__name__))
            if task == 'classification':
                pathways.setdefault(entry['route'], []).append(dict(model_id=entry['id'], status='failed'))
    return pathways, statuses
