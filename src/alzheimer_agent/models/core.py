import torch
from torch import nn
from torch.nn import functional as F
from .gated import ConvEncoder, AlzheimerUnifiedNet
from .prototypes import ROIPrototypeBank


class SpatialEncoder(ConvEncoder):
    def forward(self, x):
        grid = self.net(x)
        global_feature = self.proj(F.adaptive_avg_pool3d(grid, 1).flatten(1))
        return global_feature, grid


class DiagnosisModel(nn.Module):
    def __init__(self, route='MRI', roi_ids=(1, 2), prototypes_per_roi=3, prototype_weight=0.1):
        super().__init__()
        if route not in ('MRI', 'MRI_PET', 'MRI_TABLE', 'TABLE'):
            raise ValueError('Use one of the four registered diagnostic pathways')
        self.route, self.prototype_weight = route, prototype_weight
        modalities = ['mri', 'pet'] if route == 'MRI_PET' else ([] if route == 'TABLE' else ['mri'])
        self.encoders = nn.ModuleDict({m: SpatialEncoder() for m in modalities})
        self.banks = nn.ModuleDict({m: ROIPrototypeBank(96, 3, roi_ids, prototypes_per_roi) for m in modalities})
        self.table = nn.Sequential(nn.Linear(12, 128), nn.LayerNorm(128), nn.GELU()) if route == 'MRI_TABLE' else None
        self.global_head = nn.Linear(128 * (len(modalities)+(self.table is not None)), 3) if route != 'TABLE' else None
        self.gated = AlzheimerUnifiedNet('classification', 'TABLE_ONLY') if route == 'TABLE' else None

    def forward(self, batch):
        if self.gated is not None:

            dummy = batch['table'].new_zeros((len(batch['table']), 1, 1, 1, 1))
            return dict(logits=self.gated(dummy, dummy, batch['table']), prototypes={})
        features, explanations = [], {}
        for modality, encoder in self.encoders.items():
            global_feature, grid = encoder(batch[modality])
            features.append(global_feature)
            explanations[modality] = self.banks[modality](grid, batch[f'{modality}_atlas'])
        if self.table is not None:
            features.append(self.table(batch['table']))
        global_logits = self.global_head(torch.cat(features, dim=1))
        logits = global_logits + self.prototype_weight * sum(x['logits'] for x in explanations.values())
        return dict(logits=logits, global_logits=global_logits, prototypes=explanations)


class BrainAgeModel(nn.Module):

    def __init__(self, with_sex=True, input_shape=(192, 192, 192)):
        super().__init__()
        self.with_sex, self.input_shape = with_sex, tuple(input_shape)
        self.encoder = SpatialEncoder()
        self.project = nn.Conv3d(96, 128, 1)
        self.sex_embedding = nn.Embedding(2, 128) if with_sex else None
        self.position = nn.Parameter(torch.randn(1, 28 if with_sex else 27, 128)*0.01)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))

    def forward(self, batch):
        image = batch['mri']
        if tuple(image.shape[-3:]) != self.input_shape:
            raise ValueError('Brain age requires the declared rigid 1 mm input grid')
        _, grid = self.encoder(image)
        tokens = self.project(F.adaptive_avg_pool3d(grid, 3)).flatten(2).transpose(1, 2)
        if self.with_sex:
            sex = batch['sex']
            if not torch.all((sex == 0) | (sex == 1)):
                raise ValueError('MRI+Sex requires recorded sex (0/1); use MRI branch if missing')
            tokens = torch.cat([tokens, self.sex_embedding(sex.long())[:, None]], dim=1)
        age = self.head(self.transformer(tokens+self.position).mean(1)).flatten()
        return dict(age=age)


class ProgressionModel(nn.Module):

    def __init__(self, horizon_years, roi_ids=(1, 2), prototypes_per_roi=3, prototype_weight=0.1):
        super().__init__()
        if horizon_years not in (1, 3):
            raise ValueError('horizon_years must be 1 or 3')
        self.horizon_years, self.prototype_weight = horizon_years, prototype_weight
        self.encoder = SpatialEncoder()
        self.ordinal_references = nn.Parameter(torch.randn(2, 128)*0.1)
        self.local = ROIPrototypeBank(96, 2, roi_ids, prototypes_per_roi)
        self.head = nn.Sequential(nn.Linear(130, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, batch):
        feature, grid = self.encoder(batch['mri'])
        ordinal_distances = torch.cdist(feature.float(), self.ordinal_references.float()).square()
        local = self.local(grid, batch['mri_atlas'])
        logits = self.head(torch.cat([feature, ordinal_distances], dim=1)) + self.prototype_weight*local['logits']
        return dict(logits=logits, ordinal_distances=ordinal_distances, prototypes={'mri': local})

    def ordinal_loss(self, images, clinical_labels):

        if not torch.all((clinical_labels == 0) | (clinical_labels == 2)):
            raise ValueError('Ordinal reference learning accepts CN/AD reference samples only')
        features, _ = self.encoder(images)
        d = torch.cdist(features.float(), self.ordinal_references.float()).square()
        return F.cross_entropy(-d, (clinical_labels == 2).long())


def build_model(spec):
    kind = spec['task']
    kwargs = dict(spec.get('kwargs', {}))
    if kind == 'classification':
        return DiagnosisModel(**kwargs)
    if kind == 'brain_age':
        if spec.get('implementation') == 'author_brainmri':
            from .brainmri import AuthorBrainMRI
            return AuthorBrainMRI(**kwargs)
        return BrainAgeModel(**kwargs)
    if kind == 'progression':
        return ProgressionModel(**kwargs)
    raise ValueError('Unknown model task')
