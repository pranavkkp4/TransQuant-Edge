from pathlib import Path
import pandas as pd

def main():
    repo = Path(__file__).resolve().parents[1]
    runs = repo/"runs"/"results"
    paper = repo/"paper"

    deploy = runs/"deploy_profile.csv"
    if deploy.exists():
        # write a small TeX snippet for the deployment table
        df = pd.read_csv(deploy)
        lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
                 "Method & p50 (ms) & p95 (ms) & Peak VRAM (MB) & Size (MB)\\\\",
                 "\\midrule"]
        for _, r in df.iterrows():
            peak = "" if pd.isna(r['peak_vram_mb']) else f"{r['peak_vram_mb']:.1f}"
            size = "" if pd.isna(r['model_size_mb']) else f"{r['model_size_mb']:.1f}"
            lines.append(f"{r['method']} & {r['p50_ms']:.2f} & {r['p95_ms']:.2f} & {peak} & {size}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (paper/"deploy_table.tex").write_text("\n".join(lines) + "\n")

    metrics = runs/"method_metrics.csv"
    if metrics.exists():
        mm = pd.read_csv(metrics)
        lines = ["\\begin{tabular}{lrr}", "\\toprule",
                 "Method & Accuracy & Eval loss\\\\",
                 "\\midrule"]
        for _, r in mm.iterrows():
            acc = "" if pd.isna(r['accuracy']) else f"{float(r['accuracy']):.4f}"
            loss = "" if pd.isna(r['eval_loss']) else f"{float(r['eval_loss']):.4f}"
            lines.append(f"{r['method']} & {acc} & {loss}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (paper/"accuracy_table.tex").write_text("\n".join(lines) + "\n")

    fig_path = runs/"accuracy_latency_size.png"
    if fig_path.exists():
        rel = fig_path
        try:
            rel = fig_path.relative_to(paper)
        except ValueError:
            import os
            rel = Path(os.path.relpath(fig_path, paper))
        (paper/"figure_path.txt").write_text(str(rel) + "\n")

    # Add a short README for paper build
    (paper/"BUILD.md").write_text(
        "To build the IEEE paper:\n\n"
        "cd paper\n"
        "pdflatex paper.tex\n"
        "bibtex paper\n"
        "pdflatex paper.tex\n"
        "pdflatex paper.tex\n"
    )
    print("Paper helpers generated. Compile from paper/.")

if __name__ == "__main__":
    main()
