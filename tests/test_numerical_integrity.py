from copy import deepcopy
import pytest
from alzheimer_agent.demo import synthetic_case
from alzheimer_agent.memory import Evidence
from alzheimer_agent.agents import image_attachments


def test_bag_must_match_age_difference():
    item=deepcopy(synthetic_case()['evidence'][2]);item['content']['bag_raw']=99
    with pytest.raises(ValueError):Evidence(**item).validate()


def test_icv_percentage_must_match_soft_volume():
    item=deepcopy(synthetic_case()['evidence'][0]);item['content'].update(soft_mm3=3000.,icv_mm3=1500000.,percent_icv=3.)
    with pytest.raises(ValueError):Evidence(**item).validate()


def test_tampered_image_blocked(tmp_path):
    item=deepcopy(synthetic_case()['evidence'][0]);image=tmp_path/'image.png';image.write_bytes(b'not the recorded image')
    item['content']['images']=[dict(id='image',path='image.png',sha256='0'*64)]
    with pytest.raises(ValueError):image_attachments([item],tmp_path)


def test_mri_evidence_cannot_be_renamed_as_independent_clinical_domain():
    item=deepcopy(synthetic_case()['evidence'][2]);item['source']['domain']='clinical'
    with pytest.raises(ValueError):Evidence(**item).validate()


def test_vqa_risk_keeps_inapplicability_after_final_ad():
    from alzheimer_agent.demo import SyntheticBackend
    from alzheimer_agent.memory import MemoryBank
    from alzheimer_agent.vqa import VQAAgent
    class RiskBackend(SyntheticBackend):
        def complete(self,**kwargs):
            if kwargs['task']=='vqa':
                claim=dict(text='The stored conditional estimate is retained.',evidence_ids=['risk.3y'],
                           conflicting_evidence_ids=[],image_ids=[],literature_ids=[],
                           facts=[dict(evidence_id='risk.3y',field='value')])
                return dict(answerability='answered',claims=[claim],missing_information=[],requested_tools=[])
            return super().complete(**kwargs)
    answer=VQAAgent(RiskBackend()).ask('What is the future conversion risk?',MemoryBank.from_case(synthetic_case()),{'label':'AD'})
    assert answer['claims'][0]['resolved_facts'][0]['applicability']=='inapplicable'
    assert 'inapplicable' in answer['answer']
