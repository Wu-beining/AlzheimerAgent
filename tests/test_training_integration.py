import hashlib
import json
import torch
from alzheimer_agent.training import train,TrainingConfig
from alzheimer_agent.inference import LocalPredictor
from alzheimer_agent.pipeline import AlzheimerAgent
from alzheimer_agent.demo import SyntheticBackend,synthetic_case

torch.set_num_threads(2)


def test_train_checkpoint_inference_and_agent(tmp_path):
    rows=[]
    for i in range(12):
        rows.append(dict(patient=f'p{i}',visit='baseline',cohort='ADNI',split='TRAIN' if i<9 else 'VALIDATION',
                         label=i%3,table=dict(age=60+i,sex=i%2,education_years=12,mmse=28-i%3*4,cdrsb=i%3,apoe4=0)))
    manifest=tmp_path/'manifest.jsonl';manifest.write_text('\n'.join(json.dumps(r) for r in rows))
    spec={'task':'classification','kwargs':{'route':'TABLE'}}
    output=tmp_path/'training'
    history=train(manifest,spec,output,config=TrainingConfig(epochs=1,min_epochs=1,batch_size=8))
    assert len(history)==1
    checkpoint=output/'best.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calibration={'temperature':1.,'version':'synthetic_test_calibration'}
    predictor=LocalPredictor(checkpoint,sha,calibration)
    prediction=predictor.predict(rows[0]);assert abs(sum(prediction['probabilities'])-1)<1e-6
    case=synthetic_case();case['inputs']={'table':case['inputs']['table']}
    case['model_input']=dict(rows[0],patient='SYNTHETIC',visit='baseline')
    case['model_registry']=[dict(id='local.table',route='TABLE',checkpoint=str(checkpoint),sha256=sha,calibration=calibration)]
    result=AlzheimerAgent(SyntheticBackend()).run(case)
    assert result['routing']['primary']=='TABLE' and result['routing']['initial']['status']=='available'
