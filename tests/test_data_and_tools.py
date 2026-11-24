import json
import pytest
from alzheimer_agent.data import read_manifest, TableTransform
from alzheimer_agent.demo import SyntheticBackend, synthetic_case
from alzheimer_agent.memory import Evidence, MemoryBank
from alzheimer_agent.tools import Tool, ToolScheduler
from alzheimer_agent.vqa import VQAAgent


def test_subject_leakage_blocked(tmp_path):
    path=tmp_path/'manifest.jsonl'
    rows=[dict(patient='p',visit=str(i),cohort='ADNI',split=s) for i,s in enumerate(['TRAIN','TEST'])]
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):read_manifest(path)


def test_external_cohort_excluded(tmp_path):
    path=tmp_path/'manifest.jsonl';path.write_text(json.dumps(dict(patient='p',visit='v',cohort='AIBL',split='TRAIN')))
    with pytest.raises(ValueError):read_manifest(path)


def test_missing_tabular_features_preserved():
    transform=TableTransform.fit([dict(split='TRAIN',table={'age':60}),dict(split='TRAIN',table={'age':80})])
    result=transform.apply({'age':70})
    assert result[0]==0 and result[6]==0 and all(result[7:]==1)
    with pytest.raises(ValueError):TableTransform.fit([dict(split='TEST',table={})])


def test_tool_failure_does_not_block_independent_task():
    bank=MemoryBank.from_case(synthetic_case())
    item=dict(synthetic_case()['evidence'][1],id='new.clinical',status='candidate')
    scheduler=ToolScheduler([Tool('new',(),lambda args,snapshot:[Evidence(**item)],lambda e:(True,'QC passed'))])
    result=scheduler.run_tasks([dict(name='unknown',arguments={}),dict(name='new',arguments={})],bank)
    assert result[0]['status']=='failed' and result[1]['status']=='available'
    assert 'new.clinical' in {e['id'] for e in bank.records(verified_only=True)}


def test_vqa_tool_results_reviewed_before_refresh():
    bank=MemoryBank.from_case(synthetic_case())
    item=dict(synthetic_case()['evidence'][0],id='new.volume',status='candidate')
    scheduler=ToolScheduler([Tool('measure',(),lambda args,snapshot:[Evidence(**item)],lambda e:(True,'geometry passed'))])
    class Backend(SyntheticBackend):
        calls=0
        def complete(self,**kwargs):
            raw=super().complete(**kwargs)
            if kwargs['task']=='vqa':
                self.calls+=1
                if self.calls==1:raw['requested_tools']=[dict(name='measure',arguments={})]
            return raw
    backend=Backend()
    answer=VQAAgent(backend,tool_scheduler=scheduler).ask('What is the volume?',bank,{'label':'MCI'})
    assert backend.calls==2 and len(answer['tool_calls'])==1
    assert 'new.volume' in {e['id'] for e in bank.records(verified_only=True)}
