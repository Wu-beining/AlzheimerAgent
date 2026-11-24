from dataclasses import asdict
from copy import deepcopy
from .agents import MultiAgentAnalyst
from .memory import Evidence, MemoryBank, digest
from .routing import select_pathway
from .reporting import ReportAgent
from .vqa import VQAAgent


class AlzheimerAgent:
    def __init__(self, backend, retriever=None, tool_scheduler=None, max_rounds=3, report_revisions=2):
        self.analyst = MultiAgentAnalyst(backend, retriever, max_rounds)
        self.reporter = ReportAgent(backend, report_revisions)
        self.vqa = VQAAgent(backend, retriever, tool_scheduler)
        self.scheduler = tool_scheduler

    def run(self, case, questions=()):
        case = deepcopy(case)
        bank = MemoryBank.from_case(case)
        for name, item in case['inputs'].items():
            if item.get('status') == 'available' and (item.get('patient') != bank.patient or item.get('visit') != bank.visit):
                raise ValueError(f'Input {name} patient/visit mismatch')
        tasks = self.scheduler.run_tasks(case.get('requested_tasks', []), bank) if self.scheduler else []
        pathways = case.get('pathways', {})
        if case.get('model_registry'):
            from .inference import predict_tasks
            pathways, prediction_status = predict_tasks(case['model_registry'], case['model_input'], bank,
                task_inputs=case.get('model_inputs'), device=case.get('device', 'cpu'))
            tasks.extend(prediction_status)
        routing = select_pathway(case['inputs'], pathways)
        if routing['initial'] is not None:
            bank.add(Evidence(id='diagnosis.primary', patient=bank.patient, visit=bank.visit, task='classification',
                source=dict(version='paper-route-v1', domain='model', input_hashes={'members': digest(pathways)}),
                content=dict(route=routing['primary'], **routing['initial']), unit=None, time=bank.decision_time, status='verified'))
        decision = self.analyst.run(bank, routing, artifact_root=case.get('artifact_root', '.'))
        report = self.reporter.generate(bank, decision, artifact_root=case.get('artifact_root', '.'))
        answers = [self.vqa.ask(q, bank, decision, artifact_root=case.get('artifact_root', '.')) for q in questions]
        return dict(routing=routing, decision=decision, report=report, answers=answers,
                    task_statuses=tasks, memory=bank.snapshot())
