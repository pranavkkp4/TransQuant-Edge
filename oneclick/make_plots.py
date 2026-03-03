from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    repo = Path(__file__).resolve().parents[1]
    results = repo / "runs" / "results"
    results.mkdir(parents=True, exist_ok=True)

    metrics_path = results / "method_metrics.csv"
    deploy_path = results / "deploy_profile.csv"

    if not (metrics_path.exists() and deploy_path.exists()):
        print("Missing inputs; expected method_metrics.csv and deploy_profile.csv in runs/results.")
        return

    mm = pd.read_csv(metrics_path)
    dp = pd.read_csv(deploy_path).rename(columns={"method": "method_label"})

    # merge on method tag
    df = mm.merge(dp, left_on="method", right_on="method_tag", how="inner")
    if df.empty:
        print("No matching methods between metrics and deploy profile.")
        return

    # ensure numeric
    for col in ["accuracy", "p50_ms", "model_size_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plt.figure(figsize=(7, 5))
    plt.scatter(df["p50_ms"], df["accuracy"], s=df["model_size_mb"], alpha=0.8)
    for _, r in df.iterrows():
        label = r.get("method_label", r.get("method_tag"))
        plt.annotate(label, (r["p50_ms"], r["accuracy"]))
    plt.xlabel("Latency p50 (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency (marker size = model size MB)")
    plt.grid(True, alpha=0.3)
    out_path = results / "accuracy_latency_size.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
