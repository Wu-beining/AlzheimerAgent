from pathlib import Path
import argparse,json
import torch
from alzheimer_agent.data import read_manifest,ManifestDataset,TableTransform,collate,progression_label
from alzheimer_agent.models.core import build_model
from alzheimer_agent.calibration import TemperatureCalibration
from alzheimer_agent.preprocessing import file_hash

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['VALIDATION','CALIBRATION','TEST','OOD'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--calibration')
    a=p.parse_args();saved=torch.load(a.checkpoint,map_location=a.device,weights_only=True);spec=saved['spec']
    model=build_model(spec).to(a.device);model.load_state_dict(saved['state_dict']);model.eval()
    transform=TableTransform(**saved['table_transform']);rows=[r for r in read_manifest(a.manifest) if r['split']==a.split]
    if spec['task']=='brain_age' and a.split=='CALIBRATION':rows=[r for r in rows if r.get('clinical_label')=='CN']
    if spec['task']=='progression':
        rows=[r for r in rows if progression_label(r.get('baseline'),r.get('conversion_years'),r.get('followup_years'),spec['kwargs']['horizon_years']) is not None]
    calibration=TemperatureCalibration(**json.loads(Path(a.calibration).read_text())) if a.calibration else None
    result=[]
    with torch.no_grad():
        for row in rows:
            batch={k:v.to(a.device) for k,v in collate([ManifestDataset([row],spec,transform,labeled=False)[0]]).items()}
            prediction=model(batch)
            record={k:row[k] for k in ['patient','visit','cohort','split']}
            record['checkpoint_sha256']=file_hash(a.checkpoint)
            if spec['task']=='brain_age':
                record.update(predicted_age=float(prediction['age'][0]),age=row['age'],clinical_label=row['clinical_label'])
            else:
                logits=prediction['logits'][0].cpu().float().tolist()
                label=row['label'] if spec['task']=='classification' else progression_label(row['baseline'],row.get('conversion_years'),row['followup_years'],spec['kwargs']['horizon_years'])
                record.update(logits=logits,label=label)
                if calibration:record.update(probabilities=calibration.apply(logits).tolist(),calibration_version=calibration.version)
            result.append(record)
    with Path(a.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
