"""macOS safety guard: faiss-cpu and torch each bundle their own OpenMP runtime (libomp).

Depending on import/initialisation order the process aborts with "OMP: Error #15 ... libomp already initialized" (exit
134) or segfaults, with no Python traceback. Empirically (isolated subprocess experiments during QA, macOS arm64):
  - faiss op BEFORE torch import ........................ abort
  - torch alone imported before faiss op ................ abort
  - scikit-learn AND torch imported before any faiss op .. stable, including heavy torch ops and cross-encoder inference
The production flow only worked because sentence-transformers (which imports scikit-learn and torch) was loaded before the
first FAISS search. This guard makes that ordering explicit. It is a no-op when the packages are not installed.
"""
from __future__ import annotations


def import_torch_before_faiss() -> None:
    for name in ("sklearn", "torch"):          # order matters: scikit-learn first, then torch
        try:
            __import__(name)
        except ImportError:
            pass
