from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
from .memory import LABELS
from .retrieval import MedicalGraphRAG

ROLES = ('imaging', 'clinical_cognitive', 'genetic_progression')
PATIENT_DOMAINS = {'mri', 'pet', 'clinical', 'genetic'}
PROMPTS = Path(__file__).parent/'prompts'


def image_attachments(evidence, artifact_root):
    root = Path(artifact_root).resolve()
    images = []
    for item in evidence:
        for attachment in item['content'].get('images', []):
            path = (root/attachment['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Image must be inside the case artifact root')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != attachment['sha256']:
                raise ValueError('Image checksum mismatch')
            images.append(dict(id=attachment['id'], evidence_id=item['id'], path=str(path), sha256=actual))
    if len({i['id'] for i in images}) != len(images):
        raise ValueError('Duplicate image IDs')
    return images


def validate_opinion(raw, role, evidence, images, knowledge):
    required = {'label', 'support', 'against', 'uncertainty', 'explanation', 'images_read', 'literature_ids'}
    if not isinstance(raw, dict) or not required <= raw.keys() or raw['label'] not in (*LABELS, 'abstain'):
        raise ValueError('Invalid expert opinion schema')
    ids = {e['id'] for e in evidence}
    for key, allowed in [('support', ids), ('against', ids), ('images_read', {i['id'] for i in images}),
                         ('literature_ids', {d['id'] for d in knowledge})]:
        if not isinstance(raw[key], list) or not all(isinstance(x, str) for x in raw[key]) or not set(raw[key]) <= allowed:
            raise ValueError('Unknown evidence/image/literature citation')
    if not all(isinstance(raw[k], str) for k in ('uncertainty', 'explanation')):
        raise ValueError('Opinion explanation and uncertainty must be strings')
    if raw['label'] != 'abstain' and not raw['support']:
        raise ValueError('A diagnostic vote requires patient/model evidence support')
    return dict(raw, role=role, status='reviewed')


def decide(initial, opinions, evidence):
    counts = Counter(o['label'] for o in opinions if o['label'] in LABELS)
    if initial['label'] is not None:
        counts[initial['label']] += 1
    candidates = [c for c in LABELS if counts[c] == max(counts.values(), default=0) and counts[c] > 0]
    candidate = candidates[0] if len(candidates) == 1 else None
    records = {e['id']: e for e in evidence}
    supports = sorted({eid for o in opinions if o['label'] == candidate for eid in o['support']})

    domains = sorted({records[eid]['source']['domain'] for eid in supports
                      if records[eid]['source']['domain'] in PATIENT_DOMAINS
                      and records[eid]['task'] not in ('classification', 'progression')})
    initial_label = initial['label']
    adequately_supported = bool(domains)
    changed = candidate is not None and candidate != initial_label
    if candidate is None or not adequately_supported or (initial_label is not None and changed and len(domains) < 2):
        label = initial_label
        reason = ('no_unique_candidate' if candidate is None else 'insufficient_independent_support')
    else:
        label, reason = candidate, 'reviewed_candidate'
    return dict(label=label, initial_label=initial_label, candidate=candidate, reason=reason,
                revised=label != initial_label, votes=dict(counts), support=supports, support_domains=domains,
                model_probabilities=deepcopy(initial['probabilities']),
                conflicts=sorted({eid for o in opinions for eid in o['against']}),
                uncertainty=[o['uncertainty'] for o in opinions if o['uncertainty']],
                status='insufficient_evidence' if label is None else 'reviewed')


class MultiAgentAnalyst:
    def __init__(self, backend, retriever=None, max_rounds=3, low_confidence=0.6, margin=0.1):
        if not 1 <= max_rounds <= 10:
            raise ValueError('Discussion budget must be bounded')
        self.backend, self.retriever = backend, retriever or MedicalGraphRAG()
        self.max_rounds, self.low_confidence, self.margin = max_rounds, low_confidence, margin

    def run(self, bank, routing, *, artifact_root='.'):
        initial = routing['initial']
        if initial is None:
            result = dict(label=None, status='insufficient_evidence', reason='no_diagnostic_output', rounds=[])
            bank.record_decision(result)
            return result
        evidence = bank.records(verified_only=True)
        images = image_attachments(evidence, artifact_root)
        board, errors = [], []
        for round_index in range(self.max_rounds):

            prior = deepcopy(board)
            focus = 'Review patient evidence for CN MCI AD' if not board else 'Resolve disagreements: '+str(board[-1])
            knowledge = self.retriever.retrieve(focus)
            opinions = []
            for role in ROLES:
                payload = dict(role=role, round=round_index+1, initial=initial, evidence=deepcopy(evidence),
                               previous_rounds=prior, knowledge=knowledge, focus=focus)
                try:
                    raw = self.backend.complete(task='expert', system=(PROMPTS/'expert.txt').read_text(),
                                                payload=payload, images=images if role == 'imaging' else [])
                    opinion = validate_opinion(raw, role, evidence, images if role == 'imaging' else [], knowledge)
                except Exception as exc:
                    errors.append(dict(role=role, round=round_index+1, error=type(exc).__name__))
                    opinion = dict(role=role, label='abstain', support=[], against=[], uncertainty='Expert failed validation',
                                   explanation='', images_read=[], literature_ids=[], status='failed')
                opinions.append(opinion)
            board.append(opinions)
            decision = decide(initial, opinions, evidence)
            conflict = len({o['label'] for o in opinions if o['label'] in LABELS}) > 1 or bool(decision['conflicts'])
            low = max(initial['probabilities'].values()) < self.low_confidence or initial['probability_margin'] < self.margin
            needs_review = low or conflict or decision['candidate'] is None or decision['reason'] != 'reviewed_candidate'
            if not needs_review:
                break

        decision.update(rounds=board, errors=errors, discussion_budget=self.max_rounds,
                        unresolved=needs_review, knowledge=knowledge,
                        all_conflicts=[dict(round=i+1, role=o['role'], label=o['label'], against=o['against'])
                                       for i, group in enumerate(board) for o in group
                                       if o['against'] or o['label'] not in (decision['label'], 'abstain')])
        decision['progression_applicability'] = {
            e['id']: 'applicable' if decision['label'] == 'MCI' and e['content'].get('baseline_condition') == 'MCI'
            else 'inapplicable' for e in evidence if e['task'] == 'progression'}
        bank.record_decision(decision)
        return decision
