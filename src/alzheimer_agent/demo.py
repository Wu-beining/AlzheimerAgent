from .memory import digest


def synthetic_case():
    t = '2026-01-01T12:00:00+00:00'
    common = dict(patient='SYNTHETIC', visit='baseline', time=t, status='verified')
    def evidence(eid, task, domain, content, unit=None, spatial=None):
        source = dict(version='synthetic-fixture-v1', domain=domain, input_hashes={'fixture': digest({'synthetic': True})})
        if spatial: source['spatial'] = spatial
        return dict(id=eid, task=task, content=content, unit=unit, source=source, **common)
    inputs = {m: dict(patient='SYNTHETIC', visit='baseline', status='available', qc_passed=True,
                     acquisition_date='2026-01-01') for m in ('mri', 'table')}
    return dict(patient='SYNTHETIC', visit='baseline', decision_time=t, inputs=inputs, evidence=[
        evidence('mri.volume', 'volume', 'mri', dict(measure='hippocampus', region='hippocampus', value=3200., percent_icv=0.21), 'mm3',
                 dict(space='native', atlas='SynthSeg', atlas_version='2.0')),
        evidence('clinical.cognition', 'clinical', 'clinical', dict(measure='mmse', value=24.), 'points'),
        evidence('age', 'brain_age', 'mri', dict(predicted_age=76., chronological_age=72., bag_raw=4., bag_corrected=2., calibration_version='synthetic'), 'years'),
        evidence('risk.1y', 'progression', 'model', dict(value=0.2, horizon_years=1, baseline_condition='MCI'), 'probability'),
        evidence('risk.3y', 'progression', 'model', dict(value=0.4, horizon_years=3, baseline_condition='MCI'), 'probability')],
        pathways={'MRI_TABLE': [dict(model_id='synthetic.member', status='available', calibration_version='synthetic',
                                     probabilities=dict(CN=0.1, MCI=0.8, AD=0.1))]})


class SyntheticBackend:
    def complete(self, *, task, system, payload, images):
        if task == 'expert':
            return dict(label='MCI', support=['mri.volume', 'clinical.cognition'], against=[],
                        explanation='Synthetic control-flow opinion.', uncertainty='Synthetic example only.', images_read=[], literature_ids=[])
        if task == 'fact_check':
            return dict(checks=[dict(index=i, verdict='supported', reason='Synthetic fixture only') for i in range(len(payload['claims']))])
        claim = dict(text='The recorded measurement is available in the evidence record.',
                     evidence_ids=['mri.volume'], conflicting_evidence_ids=[], image_ids=[], literature_ids=[],
                     facts=[dict(evidence_id='mri.volume', field='value')])
        if task == 'report': return dict(diagnosis=payload['decision']['label'], claims=[claim])
        if task == 'vqa': return dict(answerability='answered', claims=[claim], missing_information=[], requested_tools=[])
        raise ValueError(task)
