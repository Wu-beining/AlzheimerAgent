from copy import deepcopy
from pathlib import Path
import json
import re
from .memory import digest
from .agents import image_attachments

PROMPTS = Path(__file__).parent/'prompts'


def fact_value(evidence, fact):

    by_id = {e['id']: e for e in evidence}
    if set(fact) != {'evidence_id', 'field'} or fact['evidence_id'] not in by_id:
        raise ValueError('Invalid fact reference')
    value = by_id[fact['evidence_id']]['content']
    for part in fact['field'].split('.'):
        if not isinstance(value, dict) or part not in value:
            raise ValueError('Unknown fact field')
        value = value[part]
    if isinstance(value, (dict, list)):
        raise ValueError('Fact pointer must reference a scalar')
    e = by_id[fact['evidence_id']]
    return dict(evidence_id=e['id'], field=fact['field'], value=value, unit=e['unit'],
                horizon_years=e['content'].get('horizon_years'), region=e['content'].get('region'))


def validate_claim(claim, evidence, images=(), knowledge=()):
    expected = {'text', 'evidence_ids', 'conflicting_evidence_ids', 'image_ids', 'literature_ids', 'facts'}
    if not isinstance(claim, dict) or set(claim) != expected or not isinstance(claim['text'], str):
        raise ValueError('Invalid claim schema')

    if re.search(r'\d', claim['text']):
        raise ValueError('Put all numeric claims in source-bound facts, not prose')
    ids = {e['id'] for e in evidence}
    for key, allowed in [('evidence_ids', ids), ('conflicting_evidence_ids', ids),
                         ('image_ids', {i['id'] for i in images}), ('literature_ids', {k['id'] for k in knowledge})]:
        values = claim[key]
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values) or not set(values) <= allowed:
            raise ValueError('Unknown citation')
    if not claim['evidence_ids'] and not claim['literature_ids']:
        raise ValueError('Unsupported claim')
    if not isinstance(claim['facts'], list):
        raise ValueError('facts must be a list')
    facts = [fact_value(evidence, f) for f in claim['facts']]
    if not {f['evidence_id'] for f in facts} <= set(claim['evidence_ids']):
        raise ValueError('Facts must also be cited')
    return dict(claim, resolved_facts=facts)


def fact_check(backend, claims, *, evidence, decision, images=(), knowledge=()):
    raw = backend.complete(task='fact_check', system=(PROMPTS/'fact_check.txt').read_text(),
        payload=dict(claims=deepcopy(claims), evidence=deepcopy(evidence), decision=decision, images=images, knowledge=knowledge), images=list(images))
    checks = raw.get('checks')
    if not isinstance(checks, list) or len(checks) != len(claims):
        raise ValueError('Fact checker must cover every claim')
    if sorted(c.get('index', -1) for c in checks) != list(range(len(claims))):
        raise ValueError('Fact check coverage mismatch')
    if any(c.get('verdict') not in ('supported', 'unsupported', 'uncertain', 'diagnosis_conflict') for c in checks):
        raise ValueError('Unknown factual verdict')
    accepted = {c['index'] for c in checks if c['verdict'] == 'supported'}
    return [c for i, c in enumerate(claims) if i in accepted], checks


class ReportAgent:
    def __init__(self, backend, revision_budget=2):
        if not 0 <= revision_budget <= 5:
            raise ValueError('Invalid report revision budget')
        self.backend, self.revision_budget = backend, revision_budget

    def generate(self, bank, decision, *, artifact_root='.'):
        snapshot = bank.snapshot()
        evidence = [e for e in snapshot['evidence'] if e['status'] == 'verified']
        knowledge = decision.get('knowledge', [])
        images = image_attachments(evidence, artifact_root)
        payload = dict(decision=decision, evidence=evidence, discussion=decision.get('rounds', []), knowledge=knowledge)
        accepted, checks, errors, referrals = [], [], [], []
        for attempt in range(self.revision_budget+1):
            try:
                draft = self.backend.complete(task='report', system=(PROMPTS/'report.txt').read_text(),
                                              payload=deepcopy(payload), images=images)
                if draft.get('diagnosis') != decision['label']:
                    referrals.append('report_diagnosis_conflict')
                    break
                claims = [validate_claim(c, evidence, images=images, knowledge=knowledge) for c in draft.get('claims', [])]
                if not claims:
                    raise ValueError('No report claims')
                accepted, checks = fact_check(self.backend, claims, evidence=evidence, decision=decision, images=images, knowledge=knowledge)
                referrals.extend(c for c in checks if c['verdict'] == 'diagnosis_conflict')
                if referrals or len(accepted) == len(claims):
                    break
                payload['revision_feedback'] = checks
            except Exception as exc:
                errors.append(type(exc).__name__)
                payload['revision_feedback'] = ['Previous draft failed validation; use only source-bound facts']
        report = dict(decision=deepcopy(decision), claims=accepted, checks=checks, errors=errors,
                      arbiter_referrals=referrals, snapshot_hash=snapshot['snapshot_hash'],
                      evidence=evidence, images=images, status='requires_arbiter_review' if referrals else ('reviewed' if accepted else 'facts_only'))

        report['clinician_report'] = render_report(report, audience='clinician')
        report['patient_summary'] = render_report(report, audience='patient')
        report['technical_appendix'] = dict(snapshot=snapshot, review=deepcopy(decision), fact_checks=checks,
                                          arbiter_referrals=referrals, generation_errors=errors)
        if bank.snapshot()['snapshot_hash'] != snapshot['snapshot_hash']:
            raise RuntimeError('Evidence changed during report generation')
        return report


def render_report(report, audience):
    decision = report['decision']
    heading = 'Clinical report' if audience == 'clinician' else 'Patient summary'
    lines = [f'# {heading}', '', f"Reviewed assessment: {decision.get('label') or 'insufficient evidence'}",
             f"Initial model assessment: {decision.get('initial_label') or 'unresolved'}", '', '## Task findings', '']
    for e in report['evidence']:
        if e['task'] in ('progression', 'brain_age'):
            applicability = decision.get('progression_applicability', {}).get(e['id'], 'applicable')
            lines.append(f"- [{e['id']}] {e['task']}: {json.dumps(e['content'], ensure_ascii=False)}; {applicability}.")
    lines += ['', '## Reviewed interpretation', '']
    for c in report['claims']:
        lines.append(c['text']+' '+''.join(f'[{key}]' for key in c['evidence_ids']))
        for f in c['resolved_facts']:
            lines.append(f"- [{f['evidence_id']}] {f['field']}: {f['value']} {f['unit'] or ''}")
    if audience == 'clinician':
        for image in report.get('images', []):
            lines.extend(['', f"![{image['id']}]({image['path']})", f"Source: [{image['evidence_id']}]"])
        lines += ['', '## Quantification and provenance', '']
        for e in report['evidence']:
            lines.append(f"- [{e['id']}] {e['task']}: {json.dumps(e['content'], ensure_ascii=False)} ({e['unit'] or 'unitless'}).")
    lines += ['', '## Disagreements and limitations', '',
              json.dumps(decision.get('all_conflicts', []), ensure_ascii=False),
              'Unresolved review: '+str(decision.get('unresolved', True)),
              'Model probabilities: '+json.dumps(decision.get('model_probabilities', {})),
              'Conditional risks describe predictions, not observed future outcomes.',
              'Missing and failed measurements are not normal findings.',
              'Additional assessment should address the unresolved evidence identified above.']
    if report['arbiter_referrals']:
        lines.append('A narrative diagnosis conflict requires Arbiter review before use.')
    return '\n'.join(lines)+'\n'
