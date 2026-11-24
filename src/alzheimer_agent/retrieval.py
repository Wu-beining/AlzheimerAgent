from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Knowledge:
    id: str
    text: str
    source_url: str
    title: str
    entities: tuple[str, ...] = ()


class MedicalGraphRAG:

    def __init__(self, documents=(), edges=()):
        self.documents = {d.id: d for d in documents}
        if len(self.documents) != len(documents) or any(not d.source_url or not d.title for d in documents):
            raise ValueError('Knowledge needs distinct IDs, source URLs and titles')
        self.neighbors = {}
        for a, b in edges:
            self.neighbors.setdefault(a, set()).add(b)
            self.neighbors.setdefault(b, set()).add(a)

    def retrieve(self, question, *, limit=5):
        tokens = set(re.findall(r'\w+', question.lower()))
        scores, entities = {}, set()
        for key, doc in self.documents.items():
            score = len(tokens & set(re.findall(r'\w+', (doc.title+' '+doc.text).lower())))
            if score:
                scores[key] = float(score)
        seeds = sorted(scores, key=scores.get, reverse=True)[:limit]
        for key in seeds:
            entities.update(self.documents[key].entities)
        expanded = entities | {b for a in entities for b in self.neighbors.get(a, ())}
        for key, doc in self.documents.items():
            if set(doc.entities) & expanded:
                scores[key] = scores.get(key, 0)+0.25
        return [self.documents[k].__dict__ for k in sorted(scores, key=scores.get, reverse=True)[:limit]]
