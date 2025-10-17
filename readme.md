# DiT-Flow (DiT + Flow Matching) for MNIST

This repository contains an implementation of a conditional DiT (Diffusion Transformer) model trained with Flow Matching for generative modeling on the MNIST dataset. The codebase includes training, inference, and model definition modules adapted for conditional generation (class labels). The implementation is minimal and targeted for easy experimentation on a single GPU or CPU.

## Contents

- `dit.py` — DiT model implementation (patch embedding, DiT blocks, final unpatchify). Supports conditioning on time and class labels.
- `flowmatching.py` — Flow Matching training utilities and a simple ODE sampler based on Euler steps. Also contains `get_train_tuple` helper to build training tuples.
- `train.py` — Training script with arguments for epochs, model size, optimizer, and checkpointing.
- `infer.py` — Inference script to sample conditional trajectories from a checkpoint and save a grid of snapshots showing progression across sampling steps.
- `checkpoints/` — Saved model checkpoints (example files are included).
- `data/` — Dataset folder; torchvision's MNIST will download data here by default.

## Requirements

Recommended environment: Python 3.8+ with a compatible PyTorch build. The project was developed with PyTorch and torchvision. Below is a minimal requirements list; pin versions as needed for reproducibility.

- Python 3.8+
- torch
- torchvision
- timm
- matplotlib
- numpy

Install with pip (example):

```powershell
pip install torch torchvision timm matplotlib numpy
```

Notes:

- If you have a CUDA GPU, install a CUDA-enabled PyTorch build from https://pytorch.org matching your CUDA version to enable GPU acceleration.
- `timm` is required because `dit.py` re-uses `Attention` and `Mlp` components from it.



## Quick start — training

1. Prepare environment and install dependencies (see Requirements).
2. (Optional) Inspect or change training hyperparameters in `train.py` or pass CLI args.

Basic train command (runs on CPU or GPU if available):

```powershell
python train.py --epochs 50 --batch_size 32 --lr 1e-4
```

Important CLI arguments available in `train.py` (defaults shown in code):

- `--epochs` (int, default 50)
- `--batch_size` (int, default 32)
- `--lr` (float, default 1e-4)
- `--gamma` (float, default 0.5) — learning rate decay factor
- `--lr_adjust_epoch` (int, default 20) — StepLR step size
- `--print_interval` (int, default 100)
- `--save_interval` (int, default 5)
- DiT model size params: `--base_channels`, `--num_layers`, `--num_heads`

Checkpoints are saved to `checkpoints/model_epoch_{EPOCH}.pth` (state dict only). The training script uses `SummaryWriter` to write TensorBoard logs to `logs/`.

## Quick start — inference (generate sample trajectories)

The `infer.py` script loads a checkpoint and runs the Flow Matching ODE sampler to produce conditional samples for labels 0..B-1 where B is `--batch_size`.

Basic inference command:

```powershell
python infer.py --ckpt checkpoints/model_epoch_50.pth --batch_size 10 --n_steps 100 --step_interval 10 --out_dir trajectories
```

Key flags in `infer.py`:

- `--ckpt` — path to a checkpoint (default `checkpoints/model_epoch_50.pth`)
- `--hidden_size` / `--num_layers` / `--num_heads` — override model architecture if needed
- `--batch_size` — number of labels to generate (also the number of rows in the saved progression grid)
- `--n_steps` — number of Euler sampling steps
- `--step_interval` — how often to snapshot intermediate images for the progression grid
- `--out_dir` — output directory for saved PNGs

`infer.py` attempts to inspect the checkpoint to guess `num_layers` and `hidden_size` and will prefer CLI overrides if you pass them.

Outputs

- The inference script writes a PNG grid showing how each sample (rows with different labels) evolves across sampling steps. Files are saved to the `--out_dir` directory with a timestamp.

## Implementation notes and behavior

- Model: `DiTModel` accepts conditioning `c` in `forward(x, c)` where `c` can be:
	- None (defaults to zero time embedding),
	- a tensor `t` with shape (B,) providing time scalars,
	- a tuple `(t, y)` where `y` are integer labels for conditioning.
- Flow Matching: `flowmatching.py` provides `get_train_tuple` to prepare training tuples (z_t, t, target, y_out) and `sample_ode` to step trajectories using an Euler integrator.
- Checkpoint loading: `infer.py` contains `inspect_checkpoint_structure` and `load_checkpoint_robust` helpers to partially load state dicts and tolerate DataParallel 'module.' prefixes and shape mismatches.

## Reproducing experiments and tips

- Use the provided `checkpoints/` if you want to skip training and directly run `infer.py`.
- For deterministic runs, seed PyTorch, Python, and NumPy (not implemented in scripts by default).
- If GPU memory is limited, reduce `--batch_size` or `--base_channels` / `--num_layers`.
- When training is unstable reduce the learning rate or lower `base_channels`.

## Troubleshooting

- Import errors for `timm` or `torchvision`: install missing packages with pip.
- If the checkpoint doesn't match the model architecture, use `--hidden_size` and `--num_layers` in `infer.py` to match the saved model, or use `infer.py`'s automated inspection to guess values.
- If CUDA is available but you still see CPU usage, ensure your PyTorch install matches your CUDA version and that `torch.cuda.is_available()` returns True.
- If training is slow on CPU, consider using a smaller model or run on a GPU.

## License and attribution

This project is a small research/experiment repository. Check the source files for any additional licensing notices. If you reuse code or models, please attribute the original author.

