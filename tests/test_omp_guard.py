"""Regression: faiss-cpu + torch each ship an OpenMP runtime; the wrong init order aborts the interpreter on macOS
(exit 134 "OMP: Error #15" / segfault). The guard in src/retrieval/vector_store.py must keep the real flow stable."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(code: str):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env, timeout=240)


def test_faiss_search_then_heavy_torch_then_real_models_do_not_abort():
    r = run("""
        import numpy as np
        from src.retrieval.vector_store import VectorStore      # guard runs here, before faiss is used
        vs = VectorStore(); vs.build(np.eye(4, dtype='float32'), list('abcd'))
        assert vs.search(np.array([1, 0, 0, 0], dtype='float32'), 2)[0][0] == 'a'
        import torch; (torch.ones(1 << 17) * 2).sum()
        from sentence_transformers import SentenceTransformer, CrossEncoder
        m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); m.encode(['x'] * 40)
        CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').predict([('q', 'd')] * 30)
        vs.search(np.array([0, 1, 0, 0], dtype='float32'), 2)
        print('OK')
    """)
    assert r.returncode == 0 and "OK" in r.stdout, (r.returncode, r.stderr[-400:])
    assert "OMP: Error" not in r.stderr


def test_guard_is_idempotent_and_safe_without_packages():
    from src.utils.omp_guard import import_torch_before_faiss
    import_torch_before_faiss(); import_torch_before_faiss()
