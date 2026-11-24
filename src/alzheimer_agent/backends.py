from typing import Protocol

class Backend(Protocol):
    def complete(self, *, task: str, system: str, payload: dict, images: list[dict]) -> dict:

        ...

class UnconfiguredBackend:
    def complete(self, **kwargs):
        raise RuntimeError("Provide a Backend implementation; live results are never simulated")
