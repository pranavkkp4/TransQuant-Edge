import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "runs" / "results" / "method_metrics.csv"

# New PEG ablation rows to upsert
NEW_ROWS = [
    {
        "method": "peg_k2_perm",
        "seed": "1000",
        "task": "qnli",
        "accuracy": "0.4946000366099213",
        "eval_loss": "0.8799395561218262",
        "checkpoint_path": str((ROOT / "runs" / "peg_k2_perm_seed1000" / "out").resolve()),
    },
    {
        "method": "peg_k4_perm",
        "seed": "1000",
        "task": "qnli",
        "accuracy": "0.8617975471352737",
        "eval_loss": "0.330047607421875",
        "checkpoint_path": str((ROOT / "runs" / "peg_k4_perm_seed1000" / "out").resolve()),
    },
]

COLS = ["method", "seed", "task", "accuracy", "eval_loss", "checkpoint_path", "timestamp"]


def read_existing(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_existing(METRICS_PATH)
    idx = {(r.get("method"), r.get("seed"), r.get("task")): r for r in rows}

    now = datetime.now().isoformat(timespec="seconds")
    for r in NEW_ROWS:
        key = (r["method"], r["seed"], r["task"])
        idx[key] = {
            "method": r["method"],
            "seed": r["seed"],
            "task": r["task"],
            "accuracy": r["accuracy"],
            "eval_loss": r["eval_loss"],
            "checkpoint_path": r["checkpoint_path"],
            "timestamp": now,
        }

    merged = list(idx.values())
    merged.sort(key=lambda x: (x["task"], x["seed"], x["method"]))
    write_rows(METRICS_PATH, merged)
    print(f"[ok] updated {METRICS_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
