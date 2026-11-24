from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Tool:
    name: str
    dependencies: tuple[str, ...]
    run: object
    review: object


class ToolScheduler:
    def __init__(self, tools=()):
        self.tools = {t.name: t for t in tools}
        self.reviews = {}

    def compute(self, request, snapshot):
        if set(request) != {'name', 'arguments'} or request['name'] not in self.tools:
            raise ValueError('Unknown tool request')
        tool = self.tools[request['name']]
        verified = {e['task'] for e in snapshot['evidence'] if e['status'] == 'verified'}
        if not set(tool.dependencies) <= verified:
            raise ValueError('Required verified evidence unavailable')
        items = tool.run(deepcopy(request['arguments']), deepcopy(snapshot))
        for item in items:
            if item.status != 'candidate':
                raise ValueError('New tool outputs must enter as candidates')
            self.reviews[item.id] = tool.review
        return items

    def review(self, item):
        return self.reviews[item.id](item)

    def run_tasks(self, requests, bank):
        results = []
        for request in requests:
            try:
                items = self.compute(request, bank.snapshot())
                for item in items:
                    bank.add(item)
                    accepted, reason = self.review(item)
                    bank.review(item.id, accepted=accepted, reviewer=request['name'], reason=reason)
                results.append(dict(name=request['name'], status='available', evidence_ids=[i.id for i in items]))
            except Exception as exc:
                results.append(dict(name=request['name'], status='failed', error=type(exc).__name__))
        return results
