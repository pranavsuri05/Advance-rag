"""CLI: run the gold-dataset evaluation against the real Gemini-backed pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import build_services  # noqa: E402
from src.evaluation.evaluator import Evaluator, load_dataset, save_results, validate_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None, help="path to dataset JSON (default: evaluation/dataset.json)")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-as-a-judge scoring (saves quota)")
    args = ap.parse_args()

    svc = build_services()
    if svc.llm is None:
        print("GEMINI_API_KEY is not set. Add it to .env first.", file=sys.stderr)
        return 2
    if not svc.kb.ready:
        print("Indexing sample corpus first...")
        svc.kb.rebuild()
    items = load_dataset(Path(args.dataset) if args.dataset else svc.settings.eval_dataset)[: args.limit]
    for problem in validate_dataset(items, {p.name for p in svc.settings.documents_dir.iterdir()}):
        print(f"dataset warning: {problem}", file=sys.stderr)
    ev = Evaluator(svc.pipeline, svc.llm, use_judge=not args.no_judge)
    run = ev.run(items, progress=lambda i, n, q: print(f"[{i + 1}/{n}] {q[:80]}" if i < n else "done"))
    path = save_results(run, svc.settings.logs_dir / "evals")
    print("\nAggregate metrics:")
    for k, v in run["aggregate"].items():
        print(f"  {k:36s} {v if v is None or isinstance(v, int) else round(v, 3)}")
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
