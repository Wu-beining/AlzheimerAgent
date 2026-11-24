from copy import deepcopy
import pytest
from alzheimer_agent.demo import SyntheticBackend, synthetic_case
from alzheimer_agent.memory import Evidence, MemoryBank
from alzheimer_agent.routing import select_pathway, vote_members
from alzheimer_agent.agents import decide, MultiAgentAnalyst
from alzheimer_agent.pipeline import AlzheimerAgent
from alzheimer_agent.reporting import validate_claim
from alzheimer_agent.vqa import VQAAgent


def member(name, p):
    return dict(model_id=name, status='available', calibration_version='cal-v1', probabilities=dict(zip(('CN','MCI','AD'), p)))


def test_member_vote_overrules_probability_mean():
    result = vote_members([member('a', [.51,.49,0]), member('b', [.51,.49,0]), member('c', [0,0,1])])
    assert result['label'] == 'CN' and result['votes']['CN'] == 2


def test_exact_tie_retained_and_probability_tie_break():
    assert vote_members([member('a',[.6,.4,0]), member('b',[.4,.6,0])])['label'] is None
    assert vote_members([member('a',[.9,.1,0]), member('b',[.4,.6,0])])['label'] == 'CN'


def test_route_priority_failure_and_pet_date_window():
    case = synthetic_case()
    case['inputs']['pet'] = dict(case['inputs']['mri'], acquisition_date='2026-02-02')
    case['pathways']['MRI_PET'] = [member('pet',[.9,.05,.05])]
    assert select_pathway(case['inputs'],case['pathways'])['primary'] == 'MRI_TABLE'
    case['inputs']['pet']['acquisition_date'] = '2026-01-31'
    assert select_pathway(case['inputs'],case['pathways'])['primary'] == 'MRI_PET'
    case['pathways']['MRI_PET'][0]['status']='failed'
    assert select_pathway(case['inputs'],case['pathways'])['primary'] == 'MRI_TABLE'


@pytest.mark.parametrize('change', ['future','patient','outcome','hash','missing_value'])
def test_evidence_rejects_invalid_records(change):
    case=synthetic_case(); e=deepcopy(case['evidence'][0])
    if change=='future': e['available_at']='2027-01-01T00:00:00+00:00'
    if change=='patient': e['patient']='OTHER'
    if change=='outcome': e['task']='followup_outcome'
    if change=='hash': e['source']['input_hashes']={'x':'not-a-hash'}
    if change=='missing_value': e['status']='missing'
    bank=MemoryBank(case['patient'],case['visit'],case['decision_time'])
    with pytest.raises(ValueError): bank.add(Evidence(**e))


def test_bank_snapshot_isolation():
    bank=MemoryBank.from_case(synthetic_case()); snapshot=bank.snapshot()
    snapshot['evidence'][0]['content']['value']=0
    assert bank.snapshot()['evidence'][0]['content']['value']==3200


def opinion(support, label='AD'):
    return dict(label=label,support=support,against=[],uncertainty='')


def test_mri_correlations_cannot_authorize_diagnostic_revision():
    case=synthetic_case(); initial=vote_members(case['pathways']['MRI_TABLE'])
    result=decide(initial,[opinion(['mri.volume','age']) for _ in range(3)],case['evidence'])
    assert result['label']=='MCI' and result['support_domains']==['mri']
    result=decide(initial,[opinion(['mri.volume','clinical.cognition']) for _ in range(3)],case['evidence'])
    assert result['label']=='AD' and result['revised']
    assert result['model_probabilities']==initial['probabilities']


def test_model_vote_always_in_first_round():
    case=synthetic_case(); bank=MemoryBank.from_case(case)
    decision=MultiAgentAnalyst(SyntheticBackend()).run(bank,select_pathway(case['inputs'],case['pathways']))
    assert len(decision['rounds'])==1
    assert decision['votes']=={'MCI':4}


def test_bounded_failed_experts_retain_initial():
    class Broken:
        def complete(self,**kwargs): raise RuntimeError('broken')
    case=synthetic_case(); bank=MemoryBank.from_case(case)
    decision=MultiAgentAnalyst(Broken(),max_rounds=2).run(bank,select_pathway(case['inputs'],case['pathways']))
    assert decision['label']=='MCI' and len(decision['rounds'])==2 and len(decision['errors'])==6


def test_progression_kept_but_inapplicable_after_revision():
    class ADBackend(SyntheticBackend):
        def complete(self,**kwargs):
            raw=super().complete(**kwargs)
            if kwargs['task']=='expert': raw['label']='AD'
            return raw
    result=AlzheimerAgent(ADBackend()).run(synthetic_case())
    assert result['decision']['label']=='AD'
    assert set(result['decision']['progression_applicability'].values())=={'inapplicable'}
    assert any(e['id']=='risk.3y' for e in result['memory']['evidence'])


def test_report_vqa_same_facts_and_no_provider():
    result=AlzheimerAgent(SyntheticBackend()).run(synthetic_case(),['What is the volume?'])
    assert result['answers'][0]['claims'][0]['resolved_facts'][0]['value']==3200
    assert 'MCI' in result['report']['clinician_report'] and 'MCI' in result['report']['patient_summary']
    assert result['report']['snapshot_hash']==result['answers'][0]['snapshot_hash']


def test_numeric_hallucination_rejected():
    raw=SyntheticBackend().complete(task='report',system='',payload={'decision':{'label':'MCI'}},images=[])['claims'][0]
    raw['text']='The volume is 9999.'
    with pytest.raises(ValueError): validate_claim(raw,synthetic_case()['evidence'])
    raw['text']='The measurement is available.';raw['facts'][0]['field']='invented'
    with pytest.raises(ValueError): validate_claim(raw,synthetic_case()['evidence'])


def test_spatial_answer_requires_images():
    case=synthetic_case();bank=MemoryBank.from_case(case)
    answer=VQAAgent(SyntheticBackend()).ask('Where is the hippocampus?',bank,{'label':'MCI'})
    assert answer['answerability']=='unanswerable'


def test_report_diagnosis_conflict_referred_without_mutation():
    class Conflict(SyntheticBackend):
        def complete(self,**kwargs):
            result=super().complete(**kwargs)
            if kwargs['task']=='report': result['diagnosis']='AD'
            return result
    result=AlzheimerAgent(Conflict()).run(synthetic_case())
    assert result['decision']['label']=='MCI'
    assert result['report']['status']=='requires_arbiter_review'
