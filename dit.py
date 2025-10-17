import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

def affine(x, shift, scale):
    # what this function does is: for each batch element b, for each channel c,
    # it applies the transformation: x_b[:, :, c] = x_b[:, :, c] * (1 + scale_b[:, c]) + shift_b[:, c]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio = 4.0):
        super(DiTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)
        self.adaLN_modulation = nn.Sequential(nn.SELU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)) 
    
    def forward(self, x, c):
        # gamma, beta, alpha for each layer is chunked from adaLN_modulation
        # equation (1) in paper is what? x = x + alpha * Attn(LN(x) * gamma + beta)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + self.attn(affine(self.norm1(x), gamma1, beta1)) * alpha1.unsqueeze(1) # (B, N, C)
        x = x + self.mlp(affine(self.norm2(x), gamma2, beta2)) * alpha2.unsqueeze(1) #
        return x
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SELU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        # the process same as in DiTBlock add affine        
        gamma, beta = self.adaLN_modulation(c).chunk(2, dim=1)
        x = affine(self.norm(x), gamma, beta)
        x = self.linear(x)
        return x  

class PatchEmbd(nn.Module):
    # Converts image to sequence of patch embeddings (2, 2)
    def __init__(self, in_channels, image_size, dim, patch_size):
        super(PatchEmbd, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        patch_tuple = (patch_size, patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.conv_proj = nn.Conv2d(in_channels = in_channels, out_channels = dim,
                                   kernel_size = patch_tuple, stride = patch_tuple,
                                   bias=False, padding_mode='zeros') 

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_proj(x)  # (B, dim, H/patch_size, W/patch_size)
        x = x.flatten(2).permute(0, 2, 1)  # (B, num_patches, dim)
        return x

def pos_embd1d(pos, dim, depth=10000):
    # pos: (num_positions,) tensor on some device
    device = pos.device
    omega = 1.0 / (depth ** (torch.arange(dim // 2, device=device, dtype=torch.float32) / dim * 2))
    pos = pos.reshape(-1, 1).to(device)
    pos = pos * omega.reshape(1, -1)
    emb_sin = torch.sin(pos)
    emb_cos = torch.cos(pos)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def time_emb(t, dim):
    # t: (B,) or (B,1) 
    device = t.device
    t = t * 10000
    freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2, device=device))
    sin_emb = torch.sin(t[:, None] / freqs)
    cos_emb = torch.cos(t[:, None] / freqs)
    emb = torch.cat([sin_emb, cos_emb], dim=-1)
    return emb

class DiTModel(nn.Module):
    def __init__(self, in_channels=1, image_size=28, hidden_size=32, patch_size=2,
                 out_channels=1, num_heads=4, mlp_ratio=4.0, num_layers=4, num_classes=10):
        super(DiTModel, self).__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_embd = PatchEmbd(in_channels, image_size, hidden_size, patch_size)
        self.dits = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.out_layer = FinalLayer(hidden_size, patch_size, out_channels)
        # label embedding for conditional generation
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, hidden_size)

    def _unpatchify(self, x, height, width, patch_size=(2,2)):
        # ensure patch_size is tuple
        if isinstance(patch_size, (list, tuple)):
            H, W = patch_size[0], patch_size[1]
        else:
            H, W = patch_size, patch_size
        bs, num_patches, patch_dim = x.shape
        in_channels = patch_dim // (H * W)
        num_patches_h = height // H
        num_patches_w = width // W
        x = x.view(bs, in_channels, num_patches_h, num_patches_w, H, W)
        x = x.transpose(3,4).contiguous()
        reconstructed_x = x.view(bs, in_channels, height, width)
        return reconstructed_x

    def forward(self, x, c=None):
        """
        c can be:
          - None: no conditioning, time defaults to zeros
          - a tensor t of shape (B,) containing time scalars
          - a tuple/list (t, y) where t is (B,) and y is (B,) integer labels
        """
        x = self.patch_embd(x)  # (B, num_patches, hidden_size)
        device = x.device
        pos = torch.arange(self.patch_embd.num_patches, device=device)
        pos_embed = pos_embd1d(pos, self.hidden_size)  # (num_patches, hidden_size)
        x = x + pos_embed.unsqueeze(0)  # (1, num_patches, hidden_size) broadcasted
        # parse conditioning
        t = None
        y = None
        if c is None:
            t = torch.zeros(x.shape[0], device=device)
        elif isinstance(c, (list, tuple)):
            t, y = c
        else:
            t = c

        # ensure tensors on correct device
        if t is not None and t.device != device:
            t = t.to(device)
        # create time embedding
        c_time = time_emb(t, self.hidden_size)  # (B, hidden_size)
        # if labels provided, add label embedding
        if y is not None:
            if y.device != device:
                y = y.to(device)
            y_emb = self.label_emb(y.long())  # (B, hidden_size)
            c = c_time + y_emb # (B, hidden_size)
        else:
            c = c_time # if not label conditioning, just use time embedding

        for dit in self.dits:
            x = dit(x, c)
        x = self.out_layer(x, c)  # (B, num_patches, patch_size*patch_size*out_channels)
        x = self._unpatchify(x, self.image_size, self.image_size, (self.patch_embd.patch_size, self.patch_embd.patch_size))
        return x

