from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime
from copy import deepcopy
import hashlib
import math
import json

LABELS = ("CN", "MCI", "AD")
DOMAINS = {"mri", "pet", "clinical", "genetic", "model", "knowledge"}
STATUSES = {"candidate", "verified", "missing", "failed", "inapplicable"}


def instant(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Evidence timestamps require a timezone")
    return date


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Evidence:
    id: str
    patient: str
    visit: str
    task: str
    source: dict
    content: dict
    unit: str | None
    time: str
    status: str = "candidate"
    available_at: str | None = None

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def validate(self):
        if not all((self.id, self.patient, self.visit, self.task)):
            raise ValueError("Missing evidence identity")
        if self.status not in STATUSES or self.source.get("domain") not in DOMAINS:
            raise ValueError("Invalid evidence status/domain")
        expected_domains = {'volume': 'mri', 'brain_age': 'mri', 'suvr': 'pet',
                            'clinical': 'clinical', 'genetic': 'genetic'}
        if self.task in expected_domains and self.source['domain'] != expected_domains[self.task]:
            raise ValueError('Evidence task/source domain mismatch')
        if not self.source.get("version") or not self.source.get("input_hashes"):
            raise ValueError("Evidence requires source version and input hashes")
        hashes = self.source["input_hashes"]
        if not isinstance(hashes, dict) or not all(isinstance(v, str) and len(v) == 64
                and all(c in '0123456789abcdef' for c in v) for v in hashes.values()):
            raise ValueError("input_hashes must contain SHA-256 values")
        instant(self.time)
        instant(self.available_at or self.time)
        digest(asdict(self))
        if self.status in {"missing", "failed"} and "value" in self.content:
            raise ValueError("Missing/failed measurements cannot have numeric values")
        if self.task in {"volume", "suvr", "prototype", "image"} and self.status == "verified":
            spatial = self.source.get("spatial", {})
            if not all(spatial.get(k) for k in ("space", "atlas", "atlas_version")):
                raise ValueError("Spatial evidence requires space and atlas provenance")
        if self.task == "progression" and self.status == "verified":
            if self.content.get("baseline_condition") != "MCI" or self.content.get("horizon_years") not in (1, 3):
                raise ValueError("Progression requires MCI condition and a 1/3-year horizon")
            if not 0 <= self.content.get("value", -1) <= 1:
                raise ValueError("Invalid progression probability")
        if self.task == "volume" and self.status == "verified":
            if self.source["spatial"]["space"] != "native":
                raise ValueError("Patient volumes must be measured in native space")
        if self.status == 'verified' and self.task == 'volume':
            c = self.content
            if 'icv_mm3' in c and 'soft_mm3' in c and 'percent_icv' in c:
                if c['icv_mm3'] <= 0 or not math.isclose(c['percent_icv'], 100*c['soft_mm3']/c['icv_mm3'], rel_tol=1e-5):
                    raise ValueError('Inconsistent ICV normalization')
        if self.status == 'verified' and self.task == 'brain_age':
            c = self.content
            if not math.isclose(c['bag_raw'], c['predicted_age']-c['chronological_age'], abs_tol=1e-6):
                raise ValueError('Inconsistent raw BAG')
        return self


class MemoryBank:
    def __init__(self, patient, visit, decision_time):
        self.patient, self.visit, self.decision_time = patient, visit, decision_time
        instant(decision_time)
        self._records, self._reviews, self._decisions = {}, {}, []

    def add(self, evidence: Evidence):
        evidence.validate()
        if evidence.patient != self.patient or evidence.visit != self.visit:
            raise ValueError("Patient/visit mismatch")
        if evidence.id in self._records:
            raise ValueError("Evidence IDs are immutable; add a new version with a new ID")
        if evidence.task in {"outcome", "ground_truth", "followup_outcome"}:
            raise ValueError("Evaluation outcomes cannot enter case reasoning")
        if max(instant(evidence.time), instant(evidence.available_at or evidence.time)) > instant(self.decision_time):
            raise ValueError("Future evidence cannot enter case reasoning")
        self._records[evidence.id] = deepcopy(asdict(evidence))

    def review(self, evidence_id, *, accepted, reviewer, reason):
        item = self._records[evidence_id]
        if item["status"] in {"missing", "failed", "inapplicable"}:
            raise ValueError("Unavailable evidence cannot be marked verified")
        if not reviewer or not reason:
            raise ValueError("Review must be attributable")
        self._reviews[evidence_id] = dict(accepted=bool(accepted), reviewer=reviewer, reason=reason)

    def records(self, verified_only=False):
        result = []
        for key, original in self._records.items():
            item = deepcopy(original)
            review = self._reviews.get(key)
            if review is not None:
                item["status"] = "verified" if review["accepted"] else "candidate"
                item["review"] = deepcopy(review)
            if not verified_only or item["status"] == "verified":
                result.append(item)
        return result

    def record_decision(self, decision):
        self._decisions.append(deepcopy(decision))

    def snapshot(self):
        result = dict(patient=self.patient, visit=self.visit, decision_time=self.decision_time,
                      evidence=self.records(), decisions=deepcopy(self._decisions))
        return dict(result, snapshot_hash=digest(result))

    @classmethod
    def from_case(cls, case):
        bank = cls(case["patient"], case["visit"], case["decision_time"])
        for item in case.get("evidence", []):
            bank.add(Evidence.from_dict(item))
        return bank


def longitudinal_difference(current, previous):

    for key in ("patient", "task", "unit"):
        if current[key] != previous[key]:
            raise ValueError("Incompatible longitudinal evidence")
    if current["content"].get("measure") != previous["content"].get("measure"):
        raise ValueError("Different measures")
    if current["status"] != "verified" or previous["status"] != "verified":
        raise ValueError("Unverified longitudinal evidence")
    if instant(previous["time"]) >= instant(current["time"]):
        raise ValueError("Historical evidence must precede current measurement")
    return current["content"]["value"] - previous["content"]["value"]
