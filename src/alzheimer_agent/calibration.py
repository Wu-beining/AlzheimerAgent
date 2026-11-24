from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class BAGCalibration:
    alpha: float
    beta: float
    version: str
    min_age: float
    max_age: float

    @classmethod
    def fit(cls, predicted, ages, labels, *, split, version):
        p, a = np.asarray(predicted, dtype=float), np.asarray(ages, dtype=float)
        if split != 'CALIBRATION' or set(labels) != {'CN'}:
            raise ValueError('BAG calibration requires independent CALIBRATION CN subjects')
        if p.shape != a.shape or len(a) < 3 or np.ptp(a) == 0 or not np.isfinite([p, a]).all():
            raise ValueError('Insufficient/invalid BAG calibration observations')
        alpha, beta = np.linalg.lstsq(np.column_stack([a, np.ones(len(a))]), p-a, rcond=None)[0]
        return cls(float(alpha), float(beta), version, float(a.min()), float(a.max()))

    def apply(self, predicted, age):
        if not np.isfinite([predicted, age]).all():
            raise ValueError('Nonfinite brain age/age')
        raw = predicted-age
        return dict(predicted_age=predicted, chronological_age=age, bag_raw=raw,
                    bag_corrected=raw-(self.alpha*age+self.beta), calibration_version=self.version,
                    age_range=[self.min_age, self.max_age], in_calibration_range=self.min_age <= age <= self.max_age)


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    version: str

    def apply(self, logits):
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Temperature must be positive')
        z = np.asarray(logits, dtype=float)/self.temperature
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite logits')
        z -= z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        return p/p.sum(axis=-1, keepdims=True)

    @classmethod
    def fit(cls, logits, labels, *, split, version):
        if split != 'CALIBRATION':
            raise ValueError('Probability calibration requires CALIBRATION data')
        from scipy.optimize import minimize_scalar
        z, y = np.asarray(logits, float), np.asarray(labels, int)
        if z.ndim != 2 or len(z) != len(y) or len(y) < 2 or y.min() < 0 or y.max() >= z.shape[1]:
            raise ValueError('Invalid calibration arrays')
        def loss(log_t):
            p = cls(float(np.exp(log_t)), version).apply(z)
            return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
        result = minimize_scalar(loss, bounds=(-4, 4), method='bounded')
        if not result.success:
            raise RuntimeError('Temperature fit failed')
        return cls(float(np.exp(result.x)), version)
