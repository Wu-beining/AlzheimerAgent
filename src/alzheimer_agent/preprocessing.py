from pathlib import Path
import csv
import hashlib
import subprocess
import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def hard_volumes(labels, affine, roi_ids):
    labels, affine = np.asarray(labels), np.asarray(affine, float)
    if labels.ndim != 3 or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('Invalid label geometry')
    voxel_volume = abs(np.linalg.det(affine[:3, :3]))
    if voxel_volume <= 0:
        raise ValueError('Singular affine')
    return {int(roi): float(np.count_nonzero(labels == roi)*voxel_volume) for roi in roi_ids}


def quantify_volumes(soft_volumes, icv, labels, affine, *, atlas_version):
    if not np.isfinite(icv) or icv <= 0 or not atlas_version:
        raise ValueError('Explicit SynthSeg ICV field and atlas version are required')
    if any(not np.isfinite(v) or v < 0 for v in soft_volumes.values()):
        raise ValueError('Invalid soft volumes')
    hard = hard_volumes(labels, affine, soft_volumes)
    return {int(roi): dict(soft_mm3=float(v), hard_mm3=hard[int(roi)],
                          percent_icv=100*float(v)/icv, icv_mm3=float(icv),
                          primary='soft', space='native', atlas='SynthSeg', atlas_version=atlas_version)
            for roi, v in soft_volumes.items()}


def read_synthseg_volumes(path, columns, *, icv_column):

    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError('Expected one subject in SynthSeg volume output')
    row = rows[0]
    return {int(roi): float(row[name]) for roi, name in columns.items()}, float(row[icv_column])


def pet_suvr(pet, labels, roi_ids, whole_cerebellum_ids):
    pet, labels = np.asarray(pet, float), np.asarray(labels)
    if pet.shape != labels.shape or pet.ndim != 3 or not np.isfinite(pet).all():
        raise ValueError('PET and native labels must share verified physical geometry')
    ref = pet[np.isin(labels, whole_cerebellum_ids)]
    if not ref.size or ref.mean() <= 0:
        raise ValueError('Whole-cerebellum reference unavailable or invalid')
    result = {}
    for roi in roi_ids:
        values = pet[labels == roi]
        result[int(roi)] = (dict(status='available', value=float(values.mean()/ref.mean()),
                                reference='whole_cerebellum', unit='SUVR') if values.size
                            else dict(status='missing', reason='ROI absent'))
    return result


def rigid_brain_age_input(image_path, reference_192, output_path):

    import nibabel as nib
    from nibabel.processing import resample_from_to
    ref = nib.load(reference_192)
    if ref.shape != (192, 192, 192) or not np.allclose(ref.header.get_zooms()[:3], 1):
        raise ValueError('Brain-age reference must be 192 cubed at 1 mm')
    source = nib.load(image_path)
    image = resample_from_to(source, (ref.shape, ref.affine), order=1)
    data = image.get_fdata(dtype=np.float32)
    mask = np.abs(data) > 1e-8
    if mask.sum() == 0 or not np.isfinite(data).all():
        raise ValueError('Empty/nonfinite brain-age image')
    values = data[mask]
    span = float(values.max()-values.min())
    if span <= 1e-8:
        raise ValueError('Constant brain-age image')
    normalized = np.zeros_like(data)
    normalized[mask] = (values-float(values.min()))/span
    nib.save(nib.Nifti1Image(normalized, ref.affine), output_path)
    return dict(path=str(output_path), sha256=file_hash(output_path), shape=list(ref.shape),
                spacing_mm=[1, 1, 1], registration='rigid_6dof', normalization='brain_mask_minmax_0_1')


def preprocessing_plan(mri, template, brain_age_reference, output, pet=None):

    out = Path(output).resolve()
    def p(name): return str(out/name)
    commands = [
        ['N4BiasFieldCorrection', '-d', '3', '-i', str(mri), '-o', p('mri_n4.nii.gz')],
        ['mri_synthstrip', '-i', p('mri_n4.nii.gz'), '-o', p('mri_brain.nii.gz'), '-m', p('brain_mask.nii.gz')],
        ['antsRegistrationSyN.sh', '-d', '3', '-f', str(template), '-m', p('mri_brain.nii.gz'), '-o', p('mni_'), '-t', 's'],
        ['antsRegistrationSyN.sh', '-d', '3', '-f', str(brain_age_reference), '-m', p('mri_brain.nii.gz'), '-o', p('age_'), '-t', 'r'],
        ['mri_synthseg', '--i', p('mri_n4.nii.gz'), '--o', p('labels_native.nii.gz'), '--robust', '--parc', '--vol', p('soft_volumes.csv'), '--qc', p('segmentation_qc.csv')],
        ['antsApplyTransforms', '-d', '3', '-i', p('labels_native.nii.gz'), '-r', str(template), '-o', p('labels_mni.nii.gz'), '-n', 'GenericLabel', '-t', p('mni_1Warp.nii.gz'), '-t', p('mni_0GenericAffine.mat')],
    ]
    if pet:
        commands.extend([
            ['antsRegistrationSyN.sh', '-d', '3', '-f', p('mri_brain.nii.gz'), '-m', str(pet), '-o', p('pet_'), '-t', 'r'],
            ['antsApplyTransforms', '-d', '3', '-i', str(pet), '-r', p('labels_native.nii.gz'), '-o', p('pet_native_labels.nii.gz'), '-n', 'Linear', '-t', p('pet_0GenericAffine.mat')],
            ['antsApplyTransforms', '-d', '3', '-i', str(pet), '-r', str(template), '-o', p('pet_mni.nii.gz'), '-n', 'Linear', '-t', p('mni_1Warp.nii.gz'), '-t', p('mni_0GenericAffine.mat'), '-t', p('pet_0GenericAffine.mat')],
        ])
    return commands


def execute_plan(commands, output):

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, command in enumerate(commands):
        try:
            with (output/f'tool_{index:02d}.log').open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            records.append(dict(command=command, status='candidate', qc='requires_review'))
        except (OSError, subprocess.CalledProcessError) as exc:
            records.append(dict(command=command, status='failed', error=type(exc).__name__))
    return records
