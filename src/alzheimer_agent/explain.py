import numpy as np


def tabular_shap(predict_probabilities, training_background, values, *, split, feature_names):
    if split != 'TRAIN':
        raise ValueError('SHAP background must be TRAIN')
    import shap
    background, values = np.asarray(training_background), np.asarray(values)
    if background.ndim != 2 or values.ndim != 2 or background.shape[1] != len(feature_names):
        raise ValueError('Tabular feature schema mismatch')
    explainer = shap.Explainer(predict_probabilities, background, feature_names=feature_names)
    return explainer(values)


def render_triplanar(image_path, output, *, voxel=None, overlay_path=None):
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image = nib.as_closest_canonical(nib.load(image_path))
    data = image.get_fdata()
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError('Invalid image')
    overlay = None
    if overlay_path:
        overlay_img = nib.as_closest_canonical(nib.load(overlay_path))
        if overlay_img.shape != image.shape or not np.allclose(overlay_img.affine, image.affine):
            raise ValueError('Overlay must share image physical geometry')
        overlay = overlay_img.get_fdata()
    voxel = voxel or tuple(n//2 for n in data.shape)
    if any(not 0 <= p < n for p, n in zip(voxel, data.shape)):
        raise ValueError('Requested voxel outside image')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (ax, title) in enumerate(zip(axes, ('Sagittal', 'Coronal', 'Axial'))):
        ax.imshow(np.rot90(np.take(data, voxel[axis], axis=axis)), cmap='gray')
        if overlay is not None:
            layer = np.rot90(np.take(overlay, voxel[axis], axis=axis))
            ax.imshow(np.ma.masked_where(layer == 0, layer), cmap='inferno', alpha=0.5)
        ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)
    return dict(path=str(output), coordinate_system='canonical_RAS', voxel=list(voxel),
                world_mm=nib.affines.apply_affine(image.affine, voxel).tolist())
