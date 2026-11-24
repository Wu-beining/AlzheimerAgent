import numpy as np
import pytest
import torch
from alzheimer_agent.models.prototypes import ROIPrototypeBank
from alzheimer_agent.models.core import DiagnosisModel, BrainAgeModel, ProgressionModel
from alzheimer_agent.calibration import BAGCalibration, TemperatureCalibration
from alzheimer_agent.data import progression_label, read_manifest, TableTransform
from alzheimer_agent.preprocessing import quantify_volumes, pet_suvr, preprocessing_plan

torch.set_num_threads(2)


def test_spatial_minimum_and_log_similarity_equations():
    bank=ROIPrototypeBank(1,2,[1,2],1)
    with torch.no_grad(): bank.prototypes.zero_()
    grid=torch.tensor([[[[[2.,9.],[3.,4.]]]]])
    atlas=torch.tensor([[[1,2],[1,2]]])
    out=bank(grid,atlas)
    assert torch.allclose(out['distances'][0,:,0,0],torch.tensor([4.,4.]))
    assert torch.allclose(out['distances'][0,:,1,0],torch.tensor([16.,16.]))
    expected=torch.log(torch.tensor(5./(4.+1e-4)))
    assert torch.allclose(out['similarity'][0,0,0,0],expected)
    assert torch.all(out['contributions']>=0)


def test_projection_restricts_training_class_and_roi():
    bank=ROIPrototypeBank(1,2,[1],1)
    grid=torch.tensor([[[[[1.,2.]]]],[[[[8.,9.]]]]])
    atlas=torch.ones(1,1,2,dtype=torch.long)
    metadata=[dict(split='TRAIN',sample_id=str(i),input_sha256='a'*64,spatial={'space':'MNI'},receptive_field={'stride':1}) for i in range(2)]
    batches=[(grid,atlas,torch.tensor([0,1]),metadata)]
    with pytest.raises(ValueError):bank.project(batches,split='TEST',modality='mri',atlas_version='AAL3')
    records=bank.project(batches,split='TRAIN',modality='mri',atlas_version='AAL3')
    assert len(records)==2 and bank.prototypes[0,0,0,0] in (1.,2.) and bank.prototypes[1,0,0,0] in (8.,9.)


@pytest.mark.parametrize('route',['MRI','MRI_PET','MRI_TABLE','TABLE'])
def test_diagnosis_forward_backward(route):
    model=DiagnosisModel(route=route,roi_ids=[1],prototypes_per_roi=1)
    batch=dict(mri=torch.randn(2,1,32,32,32),pet=torch.randn(2,1,32,32,32),table=torch.randn(2,12),
               mri_atlas=torch.ones(2,2,2,dtype=torch.long),pet_atlas=torch.ones(2,2,2,dtype=torch.long))
    out=model(batch); assert out['logits'].shape==(2,3)
    out['logits'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if route=='MRI_PET': assert model.banks['mri'].prototypes is not model.banks['pet'].prototypes


def test_horizons_separate_and_age_transformer():
    one,three=ProgressionModel(1,roi_ids=[1]),ProgressionModel(3,roi_ids=[1])
    assert one.local.prototypes is not three.local.prototypes
    batch=dict(mri=torch.randn(2,1,32,32,32),mri_atlas=torch.ones(2,2,2,dtype=torch.long))
    out=one(batch);assert out['logits'].shape==(2,2)
    one.ordinal_loss(batch['mri'],torch.tensor([0,2])).backward()
    age=BrainAgeModel(input_shape=(32,32,32))
    assert age(dict(mri=batch['mri'],sex=torch.tensor([0,1])))['age'].shape==(2,)
    with pytest.raises(ValueError): age(dict(mri=batch['mri'],sex=torch.tensor([0,2])))


def test_bag_corrects_age_dependence_without_changing_prediction():
    age=np.array([50.,60.,70.,80.]);pred=age+0.2*age-10
    cal=BAGCalibration.fit(pred,age,['CN']*4,split='CALIBRATION',version='v1')
    result=cal.apply(74.,70.)
    assert abs(result['bag_corrected'])<1e-10 and result['predicted_age']==74
    with pytest.raises(ValueError):BAGCalibration.fit(pred,age,['CN']*4,split='TEST',version='bad')


def test_temperature_fit_only_on_calibration():
    cal=TemperatureCalibration.fit([[2,0],[0,2],[1,0],[0,1]],[0,1,1,0],split='CALIBRATION',version='v1')
    assert np.allclose(cal.apply([[2,0]]).sum(1),1)
    with pytest.raises(ValueError): TemperatureCalibration.fit([[2,0],[0,2]],[0,1],split='TEST',version='bad')


def test_horizon_censoring_independent():
    assert progression_label('MCI',None,2,1)==0
    assert progression_label('MCI',None,2,3) is None
    assert progression_label('MCI',0.8,0.8,1)==1
    assert progression_label('AD',None,4,3) is None


def test_soft_volumes_icv_and_pet_reference():
    labels=np.array([[[1,1],[2,2]]]);affine=np.diag([2,3,4,1])
    result=quantify_volumes({1:40.,2:50.},1000.,labels,affine,atlas_version='2.0')
    assert result[1]['hard_mm3']==pytest.approx(48) and result[1]['percent_icv']==4
    suvr=pet_suvr(np.array([[[4,4],[2,2]]]),labels,[1],[2])
    assert suvr[1]['value']==2
    with pytest.raises(ValueError):pet_suvr(np.zeros_like(labels),labels,[1],[2])


def test_preprocessing_uses_rigid_age_generic_labels_and_pet_chain():
    commands=preprocessing_plan('mri.nii','mni.nii','age_ref.nii','out','pet.nii')
    assert any('GenericLabel' in command for command in commands)
    assert sum(c[-1]=='r' for c in commands)==2
    pet=[c for c in commands if any('pet_mni.nii' in str(x) for x in c)][0]
    assert 'mni_1Warp' in pet[-5] and 'mni_0GenericAffine' in pet[-3] and 'pet_0GenericAffine' in pet[-1]
