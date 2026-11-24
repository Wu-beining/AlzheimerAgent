from pathlib import Path
import json
import numpy as np

SPLITS = {'TRAIN', 'VALIDATION', 'CALIBRATION', 'TEST', 'OOD'}
TABLE_FIELDS = ('age', 'sex', 'education_years', 'mmse', 'cdrsb', 'apoe4')


def progression_label(baseline, conversion_years, followup_years, horizon):
    if horizon not in (1, 3) or baseline != 'MCI':
        return None
    if followup_years is None or not np.isfinite(followup_years) or followup_years < 0:
        raise ValueError('Invalid follow-up duration')
    if conversion_years is not None:
        if not np.isfinite(conversion_years) or conversion_years < 0 or conversion_years > followup_years:
            raise ValueError('Invalid observed conversion time')
        if conversion_years <= horizon:
            return 1
    return 0 if followup_years >= horizon else None


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assignments, ids = {}, set()
    for row in rows:
        if row.get('split') not in SPLITS:
            raise ValueError('Unknown split')
        key = row['cohort'], row['patient']
        if key in assignments and assignments[key] != row['split']:
            raise ValueError('All visits of a participant must share one split')
        assignments[key] = row['split']
        identity = (*key, row['visit'])
        if identity in ids:
            raise ValueError('Duplicate participant visit')
        ids.add(identity)
        if row['cohort'].upper() in {'AIBL', 'OASIS1', 'OASIS2', 'OASIS-1', 'OASIS-2'} and row['split'] != 'OOD':
            raise ValueError('External cohorts cannot enter development')
    return rows


class TableTransform:
    def __init__(self, means, scales):
        self.means, self.scales = np.array(means, float), np.array(scales, float)
        if self.means.shape != (6,) or self.scales.shape != (6,) or not np.isfinite([self.means, self.scales]).all() or (self.scales <= 0).any():
            raise ValueError('Invalid table transform')

    @classmethod
    def fit(cls, rows):
        if not rows or any(r['split'] != 'TRAIN' for r in rows):
            raise ValueError('Fit tabular preprocessing only on TRAIN')
        a = np.array([[r.get('table', {}).get(f, np.nan) if r.get('table', {}).get(f) is not None else np.nan for f in TABLE_FIELDS] for r in rows], float)
        means, scales = [], []
        for col in a.T:
            good = col[np.isfinite(col)]
            means.append(float(good.mean()) if len(good) else 0.)
            scales.append(float(good.std()) if len(good) and good.std() > 1e-8 else 1.)
        return cls(means, scales)

    def apply(self, row):
        a = np.array([row.get(f) if row.get(f) is not None else np.nan for f in TABLE_FIELDS], float)
        missing = ~np.isfinite(a)
        a = np.where(missing, self.means, a)
        return np.concatenate([(a-self.means)/self.scales, missing.astype(float)]).astype('float32')


def load_image(path):
    path = Path(path)
    if path.suffix == '.npy':
        value = np.load(path, allow_pickle=False)
    else:
        import nibabel as nib
        value = nib.load(path).get_fdata(dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError('Expected a finite preprocessed 3D volume')
    return value.astype('float32')


class ManifestDataset:
    def __init__(self, rows, spec, table_transform, *, labeled=True):
        self.rows, self.spec, self.table_transform, self.labeled = rows, spec, table_transform, labeled

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        import torch
        from torch.nn import functional as F
        row, spec = self.rows[index], self.spec
        task, kwargs = spec['task'], spec.get('kwargs', {})
        route = kwargs.get('route', 'MRI')
        batch = {}
        if route == 'MRI_PET':
            from datetime import date
            mri, pet = row['inputs']['mri'], row['inputs']['pet']
            if abs((date.fromisoformat(mri['acquisition_date'])-date.fromisoformat(pet['acquisition_date'])).days) > 30:
                raise ValueError('MRI/PET acquisition window exceeds 30 days')
        modalities = [] if route == 'TABLE' else ['mri']+(['pet'] if route == 'MRI_PET' else [])
        for modality in modalities:
            info = row['inputs'][modality]
            if info.get('qc_passed') is not True:
                raise ValueError('Input requires passed QC')
            if task == 'brain_age' and (info.get('registration') != 'rigid_6dof' or info.get('spacing_mm') != [1, 1, 1]):
                raise ValueError('Brain age accepts rigid-only 1 mm inputs')
            if task != 'brain_age' and (info.get('space') != 'MNI152NLin2009cAsym' or not info.get('spatial_verified')):
                raise ValueError('Prediction image requires verified standard-space geometry')
            image = torch.from_numpy(load_image(info['path']))[None]
            batch[modality] = image
            if task != 'brain_age':
                atlas = torch.from_numpy(load_image(info['atlas_path']))
                if tuple(atlas.shape) != tuple(image.shape[-3:]) or not info.get('atlas_version'):
                    raise ValueError('Atlas must match the verified input grid and version')
                shape = tuple((int(n)+15)//16 for n in image.shape[-3:])
                batch[f'{modality}_atlas'] = F.interpolate(atlas[None, None], size=shape, mode='nearest')[0, 0].long()
        if route in ('TABLE', 'MRI_TABLE'):
            batch['table'] = torch.from_numpy(self.table_transform.apply(row.get('table', {})))
        if task == 'brain_age':
            if kwargs.get('with_sex', True):
                batch['sex'] = torch.tensor(row['sex'], dtype=torch.long)
            if self.labeled:
                batch['target'] = torch.tensor(row['age'], dtype=torch.float32)
        elif self.labeled:
            target = row.get('label')
            if task == 'progression':
                target = progression_label(row.get('baseline'), row.get('conversion_years'), row.get('followup_years'), kwargs['horizon_years'])
                if target is None:
                    raise ValueError('Censored/inapplicable rows must be excluded for this horizon')
            batch['target'] = torch.tensor(target, dtype=torch.long)
        return batch


def collate(batch):
    import torch
    result = {}
    for key in batch[0]:
        values = [row[key] for row in batch]
        if key.endswith('_atlas'):
            if not all(torch.equal(values[0], v) for v in values):
                raise ValueError('Batch requires the same verified AAL3 atlas grid')
            result[key] = values[0]
        else:
            result[key] = torch.stack(values)
    return result
