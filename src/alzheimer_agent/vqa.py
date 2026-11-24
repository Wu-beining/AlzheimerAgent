from copy import deepcopy
from pathlib import Path
import re
from .agents import image_attachments
from .reporting import validate_claim, fact_check
from .retrieval import MedicalGraphRAG

PROMPTS = Path(__file__).parent/'prompts'


def question_type(question):
    q = question.lower()
    spatial = any(w in q for w in ('where', 'location', 'slice', 'region', '位置', '切片', '脑区', '哪里'))
    background = any(w in q for w in ('why', 'meaning', 'explain', 'guideline', '为什么', '意义', '解释', '指南'))
    future = any(w in q for w in ('convert', 'progress', 'future', '进展', '转化', '未来'))
    quantitative = any(w in q for w in ('how much', 'volume', 'risk', 'age', 'value', '体积', '多少', '风险', '脑龄', '数值'))
    return dict(spatial=spatial, background=background, future=future, quantitative=quantitative)


def relevant_evidence(question, records):
    q = question.lower()
    groups = {'volume': ('volume', '体积', '萎缩'), 'suvr': ('pet', 'suvr'),
              'brain_age': ('age', 'bag', '脑龄'), 'progression': ('risk', 'convert', 'progress', '风险', '进展', '未来'),
              'prototype': ('prototype', '原型'), 'clinical': ('cognitive', 'mmse', '认知'),
              'genetic': ('apoe', 'genetic', '基因')}
    tasks = {task for task, terms in groups.items() if any(t in q for t in terms)}
    chosen = [e for e in records if e['task'] in tasks or e['id'].lower() in q]
    return chosen or records


class VQAAgent:
    def __init__(self, backend, retriever=None, tool_scheduler=None, tool_budget=1):
        if not 0 <= tool_budget <= 5:
            raise ValueError('Invalid VQA tool budget')
        self.backend = backend
        self.retriever = retriever or MedicalGraphRAG()
        self.tool_scheduler, self.tool_budget = tool_scheduler, tool_budget

    def ask(self, question, bank, decision, *, artifact_root='.'):
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
            raise ValueError('Question must contain 1-4000 characters')
        qtype, tool_records, errors = question_type(question), [], []
        for iteration in range(self.tool_budget+1):
            snapshot = bank.snapshot()
            evidence = relevant_evidence(question, bank.records(verified_only=True))
            images = image_attachments(evidence, artifact_root) if qtype['spatial'] else []
            knowledge = self.retriever.retrieve(question) if qtype['background'] else []
            payload = dict(question=question, question_type=qtype, decision=decision,
                           evidence=evidence, knowledge=knowledge, supplied_image_ids=[i['id'] for i in images])
            try:
                raw = self.backend.complete(task='vqa', system=(PROMPTS/'vqa.txt').read_text(), payload=deepcopy(payload), images=images)
                if raw.get('answerability') not in ('answered', 'partial', 'unanswerable'):
                    raise ValueError('Invalid answerability')
                missing = raw.get('missing_information', [])
                if not isinstance(missing, list) or not all(isinstance(x, str) for x in missing):
                    raise ValueError('Invalid missing_information')
                requested = raw.get('requested_tools', [])
                if requested and self.tool_scheduler and iteration < self.tool_budget:

                    for request in requested[:self.tool_budget-len(tool_records)]:
                        new_items = self.tool_scheduler.compute(request, bank.snapshot())
                        for item in new_items:
                            bank.add(item)
                            accepted, reason = self.tool_scheduler.review(item)
                            bank.review(item.id, accepted=accepted, reviewer='tool_scheduler_review', reason=reason)
                        tool_records.append(dict(request=request, evidence_ids=[e.id for e in new_items]))
                    continue
                claims = [validate_claim(c, evidence, images, knowledge) for c in raw.get('claims', [])]
                accepted, checks = fact_check(self.backend, claims, evidence=evidence, decision=decision, images=images, knowledge=knowledge) if claims else ([], [])
                if qtype['spatial'] and not images:
                    accepted = []
                    missing.append('Verified slices/ROI/prototype images for visual inspection')
                if qtype['quantitative']:

                    accepted = [c for c in accepted if c['resolved_facts']]
                if requested and not self.tool_scheduler:
                    missing.append('Requested computation requires a configured tool scheduler')
                status = 'unanswerable' if not accepted else ('partial' if missing or len(accepted) < len(claims) else raw['answerability'])
                by_id = {e['id']: e for e in evidence}
                for claim in accepted:
                    for fact in claim['resolved_facts']:
                        if by_id[fact['evidence_id']]['task'] == 'progression':
                            fact['applicability'] = 'applicable' if decision.get('label') == 'MCI' else 'inapplicable'
                lines = []
                for claim in accepted:
                    lines.append(claim['text'])
                    lines.extend(f"[{f['evidence_id']}] {f['field']}: {f['value']} {f['unit'] or ''}; {f.get('applicability', 'recorded fact')}" for f in claim['resolved_facts'])
                if qtype['future']:
                    lines.append('Conversion risks are conditional predictions, not observed outcomes; baseline MCI must remain applicable.')
                answer = dict(answer='\n'.join(lines) if accepted else 'Insufficient verified evidence.',
                    answerability=status, claims=accepted, checks=checks, missing_information=missing,
                    supporting_evidence_ids=sorted({eid for c in accepted for eid in c['evidence_ids']}),
                    conflicting_evidence_ids=sorted({eid for c in accepted for eid in c['conflicting_evidence_ids']}),
                    tool_calls=tool_records, snapshot_hash=snapshot['snapshot_hash'], errors=errors,
                    arbiter_referrals=[c for c in checks if c['verdict'] == 'diagnosis_conflict'])
                if bank.snapshot()['snapshot_hash'] != snapshot['snapshot_hash']:
                    raise RuntimeError('Evidence changed while answering')
                return answer
            except Exception as exc:
                errors.append(type(exc).__name__)
                return dict(answer='Insufficient verified evidence.', answerability='unanswerable',
                    claims=[], supporting_evidence_ids=[], conflicting_evidence_ids=[],
                    missing_information=['Answer failed evidence validation'], errors=errors, tool_calls=tool_records)
