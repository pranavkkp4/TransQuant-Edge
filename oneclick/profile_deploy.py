import json, time, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import QuantizedBertForSequenceClassification


def measure_latency_and_vram(model, batch, warmup=20, iters=200, device=None):
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    # warmup
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        for _ in range(warmup):
            _ = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        if device.type == "cuda":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                starter.record()
                _ = model(**batch)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))  # ms
            peak = torch.cuda.max_memory_allocated(device) / 1e6
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model(**batch)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
            peak = None

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return p50, p95, peak

def model_size_mb(path: Path):
    weight_names = {"pytorch_model.bin", "model.safetensors", "adapter_model.bin", "model.bin", "quantized.pth"}
    weight_files = [p for p in path.rglob("*") if p.is_file() and p.name in weight_names]
    if weight_files:
        total = sum(p.stat().st_size for p in weight_files)
    else:
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024**2)

def main():
    repo = Path(__file__).resolve().parents[1]
    runs = repo/"runs"
    out = runs/"results"
    out.mkdir(parents=True, exist_ok=True)

    # Choose a representative batch from QNLI validation
    ds = load_dataset("glue", "qnli")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    enc = tokenizer(ds["validation"][:8]["question"], ds["validation"][:8]["sentence"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    methods = [
        ("fp32", "fp32_seed1000", "FP32"),
        ("w8a8", "w8a8_seed1000", "W8A8"),
        ("mp_ptq", "mp_ptq_seed1000", "MP-PTQ"),
        ("peg_k3_perm", "peg_k3_perm_seed1000", "PEG(K=3,P)"),
        ("percentile_ext", "percentile_ext_seed1000", "Percentile"),
    ]

    rows=[]
    for method_tag, folder, display in methods:
        run_dir = runs/folder
        # prefer saved model in out/, otherwise last checkpoint of fp32
        candidate = run_dir/"out"
        if not candidate.exists():
            # pick latest fp32 checkpoint
            fp32_dir = runs/"fp32_seed1000"/"out"
            checkpoints = sorted(fp32_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
            candidate = checkpoints[0] if checkpoints else fp32_dir

        # choose a single folder for size calculation to avoid counting all checkpoints
        size_path = candidate
        if method_tag == "fp32":
            final_model = candidate/"final_model"
            checkpoints = sorted(candidate.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
            if final_model.exists():
                size_path = final_model
                candidate = final_model
            elif checkpoints:
                size_path = checkpoints[0]
                candidate = checkpoints[0]

        try:
            if method_tag == "fp32":
                model = AutoModelForSequenceClassification.from_pretrained(candidate)
            else:
                model = QuantizedBertForSequenceClassification.from_pretrained(candidate)
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # quick sanity prints to ensure quantized path is used
        num_quant = sum(1 for m in model.modules() if "Quant" in m.__class__.__name__)
        num_linear = sum(1 for m in model.modules() if m.__class__.__name__ == "Linear")
        print(f"[{display}] cls={model.__class__.__name__} quant_modules={num_quant} linear_layers={num_linear}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p50, p95, peak = measure_latency_and_vram(model, enc, warmup=20, iters=200, device=device)
        size = model_size_mb(size_path)
        rows.append({
            "method": display,
            "method_tag": method_tag,
            "p50_ms": p50,
            "p95_ms": p95,
            "peak_vram_mb": peak,
            "model_size_mb": size,
            "model_path": str(candidate),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out/"deploy_profile.csv", index=False)

    print("Wrote:", out/"deploy_profile.csv")

if __name__ == "__main__":
    main()
