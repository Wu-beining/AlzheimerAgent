from dataclasses import asdict
from pathlib import Path
import argparse,json
from alzheimer_agent.calibration import BAGCalibration,TemperatureCalibration

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['temperature','bag'],required=True);p.add_argument('--version',required=True);a=p.parse_args()
    rows=json.loads(Path(a.input).read_text())
    if not rows or any(r['split']!='CALIBRATION' for r in rows):raise ValueError('Use only CALIBRATION predictions')
    if a.kind=='bag':
        result=BAGCalibration.fit([r['predicted_age'] for r in rows],[r['age'] for r in rows],[r['clinical_label'] for r in rows],split='CALIBRATION',version=a.version)
    else:
        result=TemperatureCalibration.fit([r['logits'] for r in rows],[r['label'] for r in rows],split='CALIBRATION',version=a.version)
    with Path(a.output).open('x') as stream:json.dump(asdict(result),stream,indent=2)
