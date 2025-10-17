import time
import os
import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from flowmatching import FlowMatching
from dit import DiTModel
import argparse

def inspect_checkpoint_structure(path, device=torch.device('cpu')):
    """
    Inspect checkpoint to guess num_layers and hidden_size when possible.
    Returns (num_layers_guess, hidden_size_guess) where guesses may be None if not found.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # strip module.
    keys = [k[len('module.'):] if k.startswith('module.') else k for k in state_dict.keys()]

    # detect dits layer indices
    layer_idxs = []
    for k in keys:
        m = re.match(r'dits\.(\d+)\.', k)
        if m:
            try:
                layer_idxs.append(int(m.group(1)))
            except Exception:
                pass
    num_layers_guess = max(layer_idxs) + 1 if layer_idxs else None

    # detect hidden size from patch_embd.conv_proj.weight shape: (dim, in_channels, ph, pw)
    hidden_size_guess = None
    for k in keys:
        if k.endswith('patch_embd.conv_proj.weight') or k.endswith('patch_embd.conv_proj.weight'):
            try:
                w = state_dict[k]
                hidden_size_guess = int(w.shape[0])
                break
            except Exception:
                pass

    return num_layers_guess, hidden_size_guess


def load_checkpoint_robust(model, path, device=torch.device('cpu')):
    """Load checkpoint into model robustly:
    - Accepts either a raw state_dict or a dict containing 'state_dict' key.
    - Strips 'module.' prefix from keys (DataParallel) if present.
    - Filters out unexpected keys and keys whose shapes don't match the model.
    - Loads with strict=False so missing keys are allowed.
    Prints a short summary of what was kept/skipped.
    """
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        # common key names used when saving full checkpoints
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # strip DataParallel prefix if present
    stripped_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        stripped_state[new_k] = v

    model_state = model.state_dict()
    filtered_state = {}
    skipped = 0
    for k, v in stripped_state.items():
        if k in model_state:
            try:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped += 1
            except Exception:
                # If shape comparison fails for any reason, skip
                skipped += 1
        else:
            skipped += 1

    kept = len(filtered_state)
    total = len(stripped_state)
    missing = len([k for k in model_state.keys() if k not in filtered_state])
    print(f"Checkpoint '{path}': kept {kept}/{total} params, skipped {skipped} unexpected/mismatched, missing {missing} model params")

    # load filtered state dict. allow missing keys
    model.load_state_dict(filtered_state, strict=False)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # CLI arguments (defaults aligned with train.py)
    parser = argparse.ArgumentParser(description='Run inference with DiT model')
    parser.add_argument('--ckpt', type=str, default='checkpoints/model_epoch_50.pth', help='path to checkpoint')
    parser.add_argument('--hidden_size', type=int, default=None, help='override hidden size (channels)')
    parser.add_argument('--num_layers', type=int, default=None, help='override number of DiT layers')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size / number of labels to generate')
    parser.add_argument('--n_steps', type=int, default=100, help='number of sampling steps')
    parser.add_argument('--step_interval', type=int, default=10, help='snapshot interval for saved grid')
    parser.add_argument('--out_dir', type=str, default='trajectories', help='output dir for images')
    args = parser.parse_args()

    # Parameters you can tweak
    NUM_LABELS = args.batch_size
    BATCH_SIZE = NUM_LABELS
    N_STEPS = args.n_steps
    STEP_INTERVAL = args.step_interval  # record a snapshot every STEP_INTERVAL steps
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # Inspect checkpoint and adjust model instantiation if possible
    ckpt_path = args.ckpt
    num_layers_guess, hidden_size_guess = None, None
    try:
        num_layers_guess, hidden_size_guess = inspect_checkpoint_structure(ckpt_path, device)
        if num_layers_guess is not None:
            print(f"Detected checkpoint num_layers={num_layers_guess}")
        if hidden_size_guess is not None:
            print(f"Detected checkpoint hidden_size={hidden_size_guess}")
    except Exception as e:
        print(f"Warning: failed to inspect checkpoint structure: {e}")

    # prefer CLI overrides, then detected values, then defaults matching train.py
    model_num_layers = args.num_layers if args.num_layers is not None else (int(num_layers_guess) if (num_layers_guess is not None and num_layers_guess > 0) else 4)
    model_hidden_size = args.hidden_size if args.hidden_size is not None else (int(hidden_size_guess) if (hidden_size_guess is not None and hidden_size_guess > 0) else 64)
    model_num_heads = args.num_heads

    model = DiTModel(in_channels=1, image_size=28, hidden_size=model_hidden_size, patch_size=2, num_layers=model_num_layers, num_heads=model_num_heads, num_classes=10).to(device)

    load_checkpoint_robust(model, ckpt_path, device)
    model.eval()
    flow_matching = FlowMatching(model)

    # Initial noise: one sample per label (one row per label). We'll condition on labels 0..NUM_LABELS-1
    z0 = torch.randn(BATCH_SIZE, 1, 28, 28, device=device)
    labels = torch.arange(0, NUM_LABELS, device=device)

    # Run sampler conditionally; sample_ode now accepts y to condition generation
    traj = flow_matching.sample_ode(z0, N=N_STEPS, y=labels)
    traj_list = list(traj)

    # Determine snapshot indices (include final step)
    indices = list(range(0, len(traj_list), STEP_INTERVAL))
    if indices[-1] != len(traj_list) - 1:
        indices.append(len(traj_list) - 1)

    n_cols = len(indices)
    # rows correspond to labels
    batch_size = traj_list[0].shape[0]
    n_rows = batch_size

    # Create grid: each row is one sample's progression across columns (time snapshots)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))


    # Normalize axes shape for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r in range(n_rows):
        for c, idx in enumerate(indices):
            img_t = traj_list[idx]
            # img_t shape: (batch, channels, H, W)
            arr = img_t[r, 0].detach().cpu().numpy()
            ax = axes[r, c]
            ax.imshow(arr, cmap='gray')
            ax.axis('off')
            if r == 0:
                ax.set_title(f'step {idx}', fontsize=8)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'trajectory_progression_bs{batch_size}_steps{N_STEPS}_int{STEP_INTERVAL}_{timestamp}.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved progression grid to {out_path}')

