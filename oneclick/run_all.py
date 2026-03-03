import os, subprocess, sys, json, time
from pathlib import Path

import yaml

def run(cmd, cwd, log_path=None):
    print("\n==>", " ".join(cmd), flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    else:
        p = subprocess.run(cmd, cwd=cwd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def record_progress(out_dir: Path, method: str, status: str):
    """Append a status row to progress.csv so results are saved incrementally."""
    progress_path = out_dir / "progress.csv"
    progress_path.parent.mkdir(exist_ok=True, parents=True)
    write_header = not progress_path.exists()
    with progress_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("timestamp,method,status\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{method},{status}\n")


def append_method_metrics(results_dir: Path, row: dict):
    """Append accuracy/loss per method into method_metrics.csv."""
    csv_path = results_dir / "method_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    cols = ["method", "seed", "task", "accuracy", "eval_loss", "checkpoint_path", "timestamp"]
    with csv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        values = [str(row.get(k, "")) for k in cols]
        f.write(",".join(values) + "\n")


def read_eval_metrics(run_dir: Path, task: str):
    """Try to read eval metrics (accuracy, loss) from JSON or TXT artifacts."""
    acc = ""
    loss = ""
    json_path = run_dir / f"eval_results_{task}.json"
    txt_path = run_dir / f"eval_results_{task}.txt"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        acc = data.get("eval_accuracy", data.get("accuracy", acc))
        loss = data.get("eval_loss", loss)
    elif txt_path.exists():
        for line in txt_path.read_text().splitlines():
            if "eval_accuracy" in line or line.strip().startswith("accuracy"):
                acc = line.split("=")[-1].strip()
            if "eval_loss" in line:
                loss = line.split("=")[-1].strip()
    return acc, loss

def main():
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((Path(__file__).parent/"config.yaml").read_text())

    out_dir = repo / cfg["project"]["out_dir"]
    out_dir.mkdir(exist_ok=True, parents=True)

    task = cfg["project"]["task"]
    model_name = cfg["project"]["model_name"]
    bs = str(cfg["project"]["batch_size"])
    max_len = str(cfg["project"]["max_seq_length"])

    seeds = cfg["project"]["seed_list"]

    logs_dir = out_dir / "logs"
    results_dir = out_dir / "results"
    metrics_csv = results_dir / "method_metrics.csv"
    if metrics_csv.exists():
        metrics_csv.unlink()

    # Common Trainer scheduling options (more frequent logging/checkpoints/eval)
    trainer_flags = [
        "--logging-steps", "50",
        "--save-steps", "500",
        "--eval-strategy", "steps",
        "--eval-steps", "500",
        "--save-model",
    ]

    def has_final_model(seed_out: Path) -> bool:
        """Detect if a seed already finished (trainer_state.json present)."""
        out_sub = seed_out / "out"
        return (out_sub / "trainer_state.json").exists()

    # Baseline training per seed
    if cfg["methods"].get("fp32", True):
        for seed in seeds:
            seed_out = out_dir / f"fp32_seed{seed}"
            seed_out.mkdir(exist_ok=True, parents=True)
            if has_final_model(seed_out):
                print(f"Skipping fp32 seed {seed}: final model already present.", flush=True)
                continue
            cmd = [
                sys.executable, "main.py", "train-baseline",
                "--cuda",
                "--seed", str(seed),
                "--task", task,
                "--model-name", model_name,
                "--output-dir", str(seed_out),
                "--max-seq-length", max_len,
                "--batch-size", bs,
                "--eval-batch-size", bs,
                "--do-eval",
                "--save-model",
            ] + trainer_flags
            run(cmd, cwd=repo)
        record_progress(out_dir, "fp32", "done")

    # If fp32 already exists, still capture its metrics for method_metrics.csv
    for seed in seeds:
        seed_out = out_dir / f"fp32_seed{seed}"
        acc, loss = read_eval_metrics(seed_out, task)
        if acc or loss:
            checkpoint_path = seed_out / "out"
            append_method_metrics(
                results_dir,
                {
                    "method": "fp32",
                    "seed": seed,
                    "task": task,
                    "accuracy": acc,
                    "eval_loss": loss,
                    "checkpoint_path": checkpoint_path,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                },
            )

    # Validation helper expects --model-path points to directory containing task subfolders.
    # We'll pass each seed's out directory as model-path.
    def validate_quantized(tag, extra_flags):
        for seed in seeds:
            seed_out = out_dir / f"{tag}_seed{seed}"
            seed_out.mkdir(exist_ok=True, parents=True)
            model_path = out_dir / f"fp32_seed{seed}" / "out"  # Qualcomm saves in out/
            log_path = logs_dir / f"{tag}_seed{seed}.log"
            cmd = [
                sys.executable, "main.py", "validate-quantized",
                "--cuda",
                "--seed", str(seed),
                "--task", task,
                "--model-path", str(model_path),
                "--output-dir", str(seed_out),
                "--eval-batch-size", bs,
                "--max-seq-length", max_len,
            ] + trainer_flags + ["--qmethod", "symmetric_uniform"] + extra_flags
            run(cmd, cwd=repo, log_path=log_path)

            # Parse eval metrics JSON emitted by main.py
            acc, eval_loss = read_eval_metrics(seed_out, task)
            checkpoint_path = seed_out / "out"
            append_method_metrics(
                results_dir,
                {
                    "method": tag,
                    "seed": seed,
                    "task": task,
                    "accuracy": acc,
                    "eval_loss": eval_loss,
                    "checkpoint_path": checkpoint_path,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                },
            )
        record_progress(out_dir, tag, "done")

    if cfg["methods"].get("w8a8", True):
        validate_quantized("w8a8", ["--n-bits", "8", "--n-bits-act", "8"])

    if cfg["methods"].get("mp_ptq", True):
        qd = cfg["mixed_precision"]["quant_dict"]
        validate_quantized("mp_ptq", ["--n-bits", "8", "--n-bits-act", "8", "--quant-dict", qd])

    if cfg["methods"].get("peg_k3_perm", True):
        qd = cfg["peg"]["quant_dict"]
        flags = cfg["peg"]["flags"].split()
        validate_quantized("peg_k3_perm", ["--n-bits", "8", "--n-bits-act", "8", "--quant-dict", qd] + flags)

    if cfg["methods"].get("percentile_ext", True):
        perc = str(cfg["extension"]["percentile"])
        validate_quantized("percentile_ext", ["--n-bits", "8", "--n-bits-act", "8", "--percentile", perc])

    # Profiling (optional) on one seed output (seed0) for speed
    if cfg.get("profiling", {}).get("enabled", True):
        run([sys.executable, "oneclick/profile_deploy.py"], cwd=repo)

    # Plots + paper helpers
    run([sys.executable, "oneclick/make_plots.py"], cwd=repo)
    run([sys.executable, "oneclick/build_paper.py"], cwd=repo)

    print("\nDone. See runs/ for results and paper/ for IEEE draft.", flush=True)

if __name__ == "__main__":
    main()
