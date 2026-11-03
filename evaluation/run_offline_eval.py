"""CLI: offline retrieval evaluation. Makes NO LLM/API calls (local models only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import build_services  # noqa: E402
from src.evaluation.evaluator import load_dataset  # noqa: E402
from src.evaluation.offline import run_offline_retrieval_eval, save_offline  # noqa: E402


def main() -> int:
    svc = build_services()
    if not svc.kb.ready:
        print("Indexing sample corpus first (local embeddings only)...")
        svc.kb.rebuild()
    run = run_offline_retrieval_eval(svc.kb, svc.reranker, load_dataset(svc.settings.eval_dataset), k=svc.settings.final_top_k)
    path = save_offline(run, svc.settings.logs_dir / "evals")
    print(f"Offline retrieval evaluation (0 LLM calls) - {run['evaluated']} answerable questions, K={run['k']}, {run['num_chunks']} chunks")
    print(f"{'variant':<20}{'P@K':>7}{'R@K':>7}{'Hit@K':>8}{'MRR':>7}")
    for v, m in run["aggregate"].items():
        print(f"{v:<20}{m['precision_at_k']:>7.3f}{m['recall_at_k']:>7.3f}{m['hit_rate_at_k']:>8.3f}{m['mrr']:>7.3f}")
    for e in run["excluded"]:
        print(f"excluded {e['id']}: {e['reason']}")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
