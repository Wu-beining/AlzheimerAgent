import torch
from torch import nn
from torch.nn import functional as F


class ROIPrototypeBank(nn.Module):
    def __init__(self, feature_dim, num_classes, roi_ids, prototypes_per_roi=3, epsilon=1e-4):
        super().__init__()
        if not roi_ids or len(set(roi_ids)) != len(roi_ids) or 0 in roi_ids:
            raise ValueError("Use distinct nonbackground atlas labels")
        self.roi_ids = tuple(roi_ids)
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.prototypes = nn.Parameter(torch.randn(num_classes, len(roi_ids), prototypes_per_roi, feature_dim)*0.1)
        self.raw_weights = nn.Parameter(torch.zeros(num_classes, len(roi_ids), prototypes_per_roi))

    def forward(self, grid, atlas):

        if atlas.shape != grid.shape[-3:]:
            raise ValueError("Atlas/feature grid mismatch; map labels using nearest-neighbor first")
        features = grid.flatten(2).transpose(1, 2)
        distances, locations = [], []
        for r, roi in enumerate(self.roi_ids):
            indices = (atlas.flatten() == roi).nonzero().flatten()
            if not len(indices):
                raise ValueError(f"Required ROI {roi} absent from feature grid")
            local = features[:, indices].float()
            protos = self.prototypes[:, r].flatten(0, 1).float()
            d = torch.cdist(local, protos[None].expand(len(grid), -1, -1), p=2).square()
            minimum, where = d.min(dim=1)
            distances.append(minimum.reshape(len(grid), self.num_classes, -1))
            locations.append(indices[where].reshape(len(grid), self.num_classes, -1))
        distance = torch.stack(distances, dim=2)
        similarity = torch.log((distance+1)/(distance+self.epsilon))
        contributions = F.softplus(self.raw_weights)[None] * similarity
        return dict(distances=distance, similarity=similarity, contributions=contributions,
                    logits=contributions.sum((2, 3)), locations=torch.stack(locations, dim=2))

    def regularization(self, output, labels, margin=1.0):
        d = output['distances']
        same = F.one_hot(labels, self.num_classes).bool()[:, :, None, None].expand_as(d)
        cluster = d.masked_fill(~same, float('inf')).flatten(1).min(1).values.mean()
        other = d.masked_fill(same, float('inf')).flatten(1).min(1).values
        separation = F.relu(margin-other).mean()
        p = F.normalize(self.prototypes, dim=-1)
        similarities = p @ p.transpose(-1, -2)
        eye = torch.eye(p.shape[-2], device=p.device, dtype=torch.bool)
        diversity = similarities[..., ~eye].square().mean() if p.shape[-2] > 1 else p.sum()*0
        return dict(cluster=cluster, separation=separation, diversity=diversity)

    @torch.no_grad()
    def project(self, batches, *, split, modality, atlas_version):

        if split != 'TRAIN':
            raise ValueError("Prototype projection is restricted to TRAIN")
        best = torch.full(self.prototypes.shape[:-1], float('inf'), device=self.prototypes.device)
        replacement = self.prototypes.detach().clone()
        records = {}
        for grid, atlas, labels, metadata in batches:
            output = self(grid, atlas)
            flat = grid.flatten(2).transpose(1, 2)
            for b, label in enumerate(labels.tolist()):
                meta = metadata[b]
                if meta.get('split') != 'TRAIN' or not all(meta.get(k) for k in ('sample_id', 'input_sha256', 'spatial', 'receptive_field')):
                    raise ValueError('Projection requires TRAIN source and spatial provenance')
                for r, roi in enumerate(self.roi_ids):
                    for k in range(self.prototypes.shape[2]):
                        d = output['distances'][b, label, r, k]
                        if d < best[label, r, k]:
                            pos = int(output['locations'][b, label, r, k])
                            best[label, r, k] = d
                            replacement[label, r, k] = flat[b, pos]
                            records[(label, r, k)] = dict(modality=modality, label=label, roi=roi,
                                prototype=k, atlas_version=atlas_version, feature_index=pos, **meta)
        if not torch.isfinite(best).all():
            raise ValueError('TRAIN has no valid projection for some class/ROI prototypes')
        self.prototypes.copy_(replacement)
        return [records[key] for key in sorted(records)]
