"""
Quick outlier stats on a small validation slice to quantify heavy tails without expensive processing.

Outputs JSON with per-layer kurtosis, mean variance, and top-1% energy share.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def kurtosis(x: torch.Tensor) -> float:
    # Fisher-style kurtosis (excess not subtracted).
    x = x.float()
    m = x.mean()
    v = x.var(unbiased=False)
    if v == 0:
        return 0.0
    return float(((x - m) ** 4).mean() / (v * v))


def top1_energy_fraction(x: torch.Tensor) -> float:
    # Fraction of L2 energy contributed by top 1% magnitudes.
    x = x.float().abs().flatten()
    if x.numel() == 0:
        return 0.0
    thresh = torch.quantile(x, 0.99)
    energy_total = torch.sum(x * x)
    energy_top = torch.sum((x >= thresh) * (x * x))
    if energy_total == 0:
        return 0.0
    return float(energy_top / energy_total)


def collect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, output_hidden_states=True
    ).to(device)
    model.eval()

    ds = load_dataset("glue", args.task)["validation"][: args.samples]
    batch_enc = tokenizer(
        ds["question"], ds["sentence"], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )

    stats = []
    with torch.no_grad():
        for start in range(0, args.samples, args.batch_size):
            end = min(start + args.batch_size, args.samples)
            batch = {k: v[start:end].to(device) for k, v in batch_enc.items()}
            outputs = model(**batch)
            hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1,...)
            if not stats:
                stats = [dict(sum_var=0.0, sum_kurt=0.0, sum_top1=0.0, count=0) for _ in hidden_states]
            for i, h in enumerate(hidden_states):
                flat = h.flatten()
                stats[i]["sum_var"] += float(flat.var(unbiased=False))
                stats[i]["sum_kurt"] += kurtosis(flat)
                stats[i]["sum_top1"] += top1_energy_fraction(flat)
                stats[i]["count"] += 1

    results = []
    for i, s in enumerate(stats):
        c = max(s["count"], 1)
        results.append(
            {
                "layer_index": i,  # 0 = embeddings, 1-12 encoder blocks
                "mean_variance": s["sum_var"] / c,
                "mean_kurtosis": s["sum_kurt"] / c,
                "top1_energy_frac": s["sum_top1"] / c,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="qnli")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    parser.add_argument("--out", type=str, default="runs/results/outlier_stats_fp32.json")
    args = parser.parse_args()

    results = collect(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path} with {len(results)} layer entries.")


if __name__ == "__main__":
    main()
