# REPRODUCIBILITY CONTRACT

Fill these in after you run experiments.

- Repo: Qualcomm AI Research transformer-quantization (local fork)
- Commit hash: <fill>
- OS: <fill>
- Python: <fill>
- PyTorch: <fill>
- CUDA: <fill>
- GPU: <fill>

## Datasets / checkpoints
- GLUE data downloaded via HuggingFace `datasets`
- Pretrained model: `bert-base-uncased` (HF)

## Seed policy
We report medians across seeds:
- Night 1: 3 seeds (1000, 1001, 1002)
- Night 2: 5 seeds (1000..1004)

## How to run
- `./run_oneclick.sh` (Linux/macOS) or `run_oneclick.bat` (Windows)

## Outputs
- `runs/results/deploy_profile.csv`
- `runs/results/accuracy_latency_size.png`
- paper sources in `paper/`
