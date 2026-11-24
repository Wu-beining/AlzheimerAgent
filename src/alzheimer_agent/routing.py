from collections import Counter
from datetime import date
import math
from .memory import LABELS

PRIORITY = ("MRI_PET", "MRI_TABLE", "MRI", "TABLE")
REQUIRES = {"MRI_PET": {"mri", "pet"}, "MRI_TABLE": {"mri", "table"},
            "MRI": {"mri"}, "TABLE": {"table"}}


def availability(inputs):
    valid = {key for key, item in inputs.items() if item.get("status") == "available" and item.get("qc_passed") is True}
    if "pet" in valid:
        if "mri" not in valid:
            valid.remove("pet")
        else:
            try:
                mri, pet = inputs["mri"], inputs["pet"]
                matched = all(mri[k] == pet[k] for k in ("patient", "visit"))
                delta = abs((date.fromisoformat(mri["acquisition_date"]) - date.fromisoformat(pet["acquisition_date"])).days)
                if not matched or delta > 30:
                    valid.remove("pet")
            except (KeyError, ValueError):
                valid.remove("pet")
    return valid


def vote_members(members):
    valid, rejected = [], []
    for member in members:
        probabilities = member.get("probabilities", {})
        if (member.get("status") != "available" or not member.get("calibration_version")
                or set(probabilities) != set(LABELS)
                or any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > 1e-6):
            rejected.append(member.get("model_id", "unknown"))
            continue
        if any(v["model_id"] == member["model_id"] for v in valid):
            raise ValueError("Duplicate model member")

        valid.append(dict(member, label=max(LABELS, key=probabilities.get)))
    if not valid:
        return dict(status="unavailable", label=None, rejected=rejected)
    counts = Counter(m["label"] for m in valid)
    mean = {c: sum(m["probabilities"][c] for m in valid)/len(valid) for c in LABELS}
    tied = [c for c in LABELS if counts[c] == max(counts.values())]
    best = max(mean[c] for c in tied)
    tied = [c for c in tied if math.isclose(mean[c], best, abs_tol=1e-10, rel_tol=0)]
    ordered = sorted(mean.values(), reverse=True)
    return dict(status="available", label=tied[0] if len(tied) == 1 else None,
                tied_classes=tied if len(tied) > 1 else [], votes=dict(counts), probabilities=mean,
                probability_margin=ordered[0]-ordered[1], entropy=-sum(p*math.log(p) for p in mean.values() if p),
                agreement=max(counts.values())/len(valid), members=valid, rejected=rejected)


def select_pathway(inputs, pathways):
    present = availability(inputs)
    results = {route: vote_members(pathways.get(route, [])) if REQUIRES[route] <= present
               else dict(status="unavailable", label=None, reason="missing_or_failed_input") for route in PRIORITY}
    primary = next((r for r in PRIORITY if results[r]["status"] == "available"), None)
    return dict(primary=primary, initial=results[primary] if primary else None,
                pathways=results, auxiliary=[r for r in PRIORITY if r != primary and results[r]["status"] == "available"])
