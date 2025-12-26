"""
Conditional Diffusion Model for Variogram Parameter Conditioning
(Zero-Nugget Version - NO RECONSTRUCTION LOSS)
=============================================================

Architecture:
- Input 1: Sparse well map (128x128) where wells are marked with their values
- Input 2: Variogram parameters [Range, Aniso, Azimuth]
- Output: Full field realization (128x128)

Key Features:
1. Sparse conditioning: Wells can be any number, any location
2. Parameter conditioning: Generates fields respecting input variogram parameters
3. CPG/CFG: Conditional Parameter Generation / Classifier Free Guidance for both wells and parameters
4. NO RECONSTRUCTION LOSS: Model learns conditioning inherently through diffusion process
5. Resumable Training: Can load last checkpoint and continue training.

Training Data:
- Full field realizations from your variogram dataset
- Randomly sample wells during training for robustness
"""

import os
import math
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
from functools import partial
import csv
import argparse

# Set working directory if needed
os.chdir('E:/GAT') 


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Data
    IMG_SIZE = 128
    CHANNELS = 1
    DATA_SCALE = 3.5
    
    # Model
    DIM = 32                  # Base channel dimension
    DIM_MULTS = (1, 2, 4)    # Channel multipliers per resolution
    NUM_RES_BLOCKS = 1          # ResNet blocks per resolution
    ATTN_RESOLUTIONS = (8,)  # Apply attention at these resolutions
    DROPOUT = 0.1
    
    # Diffusion
    TIMESTEPS = 200
    BETA_START = 1e-4
    BETA_END = 0.02
    BETA_SCHEDULE = 'cosine'    # 'linear' or 'cosine'
    
    # Training
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 500
    EMA_DECAY = 0.9999
    GRAD_CLIP = 1.0
    
    # Well sampling during training
    MIN_WELLS = 50
    MAX_WELLS = 800
    
    # Parameter ranges (for normalization)
    # NUGGET REMOVED
    PARAM_RANGES = {
        'range': (100, 600),
        'aniso': (1.0, 3.0),
        'azimuth': (0, 180)
    }

    # Conditioning Strategy
    COND_DROP_PROB = 0.15 
    WELL_DROP_PROB = 0.15
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    DATA_PATH = 'variogram_realizations_FULL25.npy'
    SAVE_DIR = 'diffusion_checkpointsNONUGGETCONDV32_NO_RECON'
    
config = Config()
os.makedirs(config.SAVE_DIR, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def find_latest_checkpoint(save_dir):
    """Finds the checkpoint file with the highest epoch number."""
    # Look for files matching 'checkpoint_epoch*.pth'
    checkpoints = glob.glob(os.path.join(save_dir, "checkpoint_epoch*.pth"))
    
    if not checkpoints:
        return None
        
    # Extract epoch numbers and find the max
    latest_checkpoint = None
    max_epoch = -1
    
    for cp in checkpoints:
        # Regex to find the number after 'epoch'
        match = re.search(r'epoch(\d+)', cp)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                latest_checkpoint = cp
                
    return latest_checkpoint

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_schedule(schedule_type, timesteps, beta_start=1e-4, beta_end=0.02):
    if schedule_type == 'linear':
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == 'cosine':
        return cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings using sinusoidal encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Basic convolutional block with GroupNorm and SiLU"""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResnetBlock(nn.Module):
    """ResNet block with global embedding injection"""
    def __init__(self, in_channels, out_channels, global_emb_dim, groups=8, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(global_emb_dim, out_channels * 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, global_emb):
        h = self.act(self.norm1(self.conv1(x)))
        
        # Add global embedding (scale and shift)
        global_emb = self.time_mlp(global_emb)
        global_emb = rearrange(global_emb, 'b c -> b c 1 1')
        scale, shift = global_emb.chunk(2, dim=1)
        h = h * (scale + 1) + shift
        
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels, num_heads=4, groups=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        k = rearrange(k, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        v = rearrange(v, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        
        # Attention
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=h, w=w)
        
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# =============================================================================
# ENCODERS
# =============================================================================

class WellConditioningEncoder(nn.Module):
    """
    Encodes sparse well observations into a conditioning signal.
    
    Input: (B, 2, H, W) where:
        - Channel 0: Well values (masked by well locations)
        - Channel 1: Well mask (1 where well exists, 0 elsewhere)
    
    Output: Multi-scale feature maps for U-Net conditioning
    """
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        
        # Initial conv
        self.init_conv = nn.Conv2d(2, dim, 7, padding=3)
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        channels = [dim]
        in_ch = dim
        
        for mult in dim_mults:
            out_ch = dim * mult
            self.encoders.append(nn.Sequential(
                ConvBlock(in_ch, out_ch),
                ConvBlock(out_ch, out_ch),
            ))
            self.downsamples.append(Downsample(out_ch))
            channels.append(out_ch)
            in_ch = out_ch
        
        self.channels = channels
    
    def forward(self, x):
        """Returns list of feature maps at each resolution"""
        features = []
        
        h = self.init_conv(x)
        features.append(h)
        
        for encoder, downsample in zip(self.encoders, self.downsamples):
            h = encoder(h)
            features.append(h)
            h = downsample(h)
        
        return features


class VariogramLabelEmbedder(nn.Module):
    """
    Learns embeddings for [Range, Aniso, SinAzi, CosAzi].
    Includes a LEARNED NULL TOKEN to handle "unknown parameters"
    without zeroing out inputs (which would break sin/cos logic).
    """
    def __init__(self, input_dim=4, embed_dim=128):
        super().__init__()
        
        # MLP for transforming valid labels
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # The learnable "Null" embedding
        self.null_embedding = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, labels, mask_drop=None):
        """
        labels: (B, 4)
        mask_drop: (B,) boolean tensor. True = Drop (Use Null).
        """
        embeddings = self.mlp(labels)
        
        if mask_drop is not None:
            # Broadcast null embedding to batch size
            null_expanded = self.null_embedding.expand(embeddings.shape[0], -1)
            # Reshape mask for broadcasting
            mask_drop = mask_drop.view(-1, 1)
            # Select between standard embedding and null embedding
            embeddings = torch.where(mask_drop, null_expanded, embeddings)
            
        return embeddings


# =============================================================================
# CONDITIONAL U-NET
# =============================================================================

class ConditionalUNet(nn.Module):
    """
    U-Net for denoising with sparse well conditioning and parameter conditioning.
    
    Inputs:
        - x: Noisy image (B, 1, H, W)
        - t: Timestep (B,)
        - cond: Well conditioning (B, 2, H, W)
        - labels: Variogram parameters (B, 4)
    
    Outputs:
        - noise_pred: Predicted noise (B, 1, H, W)
    """
    def __init__(self, config):
        super().__init__()
        
        dim = config.DIM
        dim_mults = config.DIM_MULTS
        num_res_blocks = config.NUM_RES_BLOCKS
        attn_resolutions = config.ATTN_RESOLUTIONS
        dropout = config.DROPOUT
        img_size = config.IMG_SIZE
        
        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Label embedding
        self.label_emb = VariogramLabelEmbedder(input_dim=4, embed_dim=time_dim)
        
        # Conditioning encoder
        self.cond_encoder = WellConditioningEncoder(dim, dim_mults)
        
        # Initial conv (input + conditioning at full resolution)
        self.init_conv = nn.Conv2d(1 + dim, dim, 3, padding=1)
        
        # Build channel list
        channels = [dim]
        ch = dim
        for mult in dim_mults:
            channels.append(dim * mult)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        resolution = img_size
        in_ch = dim
        
        for i, mult in enumerate(dim_mults):
            out_ch = dim * mult
            
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                # Add conditioning channels
                blocks.append(ResnetBlock(in_ch + channels[i+1], out_ch, time_dim, dropout=dropout))
                in_ch = out_ch
                
                if resolution in attn_resolutions:
                    attns.append(AttentionBlock(out_ch))
                else:
                    attns.append(nn.Identity())
            
            self.encoder_blocks.append(blocks)
            self.encoder_attns.append(attns)
            
            if i < len(dim_mults) - 1:
                self.downsamples.append(Downsample(out_ch))
            else:
                self.downsamples.append(nn.Identity())
            
            resolution //= 2
        
        # Bottleneck
        mid_ch = dim * dim_mults[-1]
        self.mid_block1 = ResnetBlock(mid_ch, mid_ch, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResnetBlock(mid_ch, mid_ch, time_dim, dropout=dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(dim_mults))):
            out_ch = dim * mult
            
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            
            for j in range(num_res_blocks + 1):
                skip_ch = channels[i + 1] if j == 0 else 0
                blocks.append(ResnetBlock(in_ch + skip_ch, out_ch, time_dim, dropout=dropout))
                in_ch = out_ch
                
                resolution = img_size // (2 ** i)
                if resolution in attn_resolutions:
                    attns.append(AttentionBlock(out_ch))
                else:
                    attns.append(nn.Identity())
            
            self.decoder_blocks.append(blocks)
            self.decoder_attns.append(attns)
            
            if i > 0:
                self.upsamples.append(Upsample(out_ch))
            else:
                self.upsamples.append(nn.Identity())
        
        # Output
        self.final_block = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, 1, 3, padding=1)
        )
    
    def forward(self, x, t, cond, labels, mask_drop=None):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Label embedding
        l_emb = self.label_emb(labels, mask_drop)
        
        # Combine embeddings (simple addition)
        global_emb = t_emb + l_emb
        
        # Encode conditioning
        cond_features = self.cond_encoder(cond)
        
        # Initial conv with conditioning
        h = torch.cat([x, cond_features[0]], dim=1)
        h = self.init_conv(h)
        
        # Encoder with skip connections
        skips = []
        cond_idx = 1
        
        for blocks, attns, downsample in zip(self.encoder_blocks, self.encoder_attns, self.downsamples):
            for block, attn in zip(blocks, attns):
                # Concatenate conditioning at this resolution
                cond_feat = cond_features[cond_idx]
                if cond_feat.shape[2:] != h.shape[2:]:
                    cond_feat = F.interpolate(cond_feat, size=h.shape[2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, cond_feat], dim=1)
                h = block(h, global_emb)
                h = attn(h)
            skips.append(h)
            h = downsample(h)
            cond_idx = min(cond_idx + 1, len(cond_features) - 1)
        
        # Bottleneck
        h = self.mid_block1(h, global_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, global_emb)
        
        # Decoder
        for blocks, attns, upsample in zip(self.decoder_blocks, self.decoder_attns, self.upsamples):
            for i, (block, attn) in enumerate(zip(blocks, attns)):
                if i == 0 and len(skips) > 0:
                    skip = skips.pop()
                    if skip.shape[2:] != h.shape[2:]:
                        skip = F.interpolate(skip, size=h.shape[2:], mode='bilinear', align_corners=False)
                    h = torch.cat([h, skip], dim=1)
                h = block(h, global_emb)
                h = attn(h)
            h = upsample(h)
        
        # Output
        noise_pred = self.final_block(h)
        
        return noise_pred


# =============================================================================
# DIFFUSION MODEL
# =============================================================================

class ConditionalDiffusion(nn.Module):
    """
    Conditional Diffusion Model for variogram parameter conditioning.
    NO RECONSTRUCTION LOSS - learns conditioning through diffusion process only.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # U-Net
        self.model = ConditionalUNet(config)
        
        # Setup diffusion schedule
        betas = get_schedule(config.BETA_SCHEDULE, config.TIMESTEPS, 
                            config.BETA_START, config.BETA_END)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, cond, labels, t=None):
        """
        Compute training losses - NOISE PREDICTION ONLY.
        
        Args:
            x_start: Clean images (B, 1, H, W)
            cond: Well conditioning (B, 2, H, W)
            labels: Variogram parameters (B, 4)
            t: Timesteps (optional, sampled if None)
        """
        b = x_start.shape[0]
        device = x_start.device
        
        if t is None:
            t = torch.randint(0, self.config.TIMESTEPS, (b,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # --- INDEPENDENT DROPOUT LOGIC ---
        
        # 1. Mask for Labels (Soft Params) - True = Drop (Use Null Token)
        mask_drop_labels = torch.bernoulli(torch.full((b,), self.config.COND_DROP_PROB, device=device)).bool()
        
        # 2. Mask for Wells (Soft Spatial) - True = Drop (Zero inputs)
        mask_drop_wells = torch.bernoulli(torch.full((b,), self.config.WELL_DROP_PROB, device=device)).bool()
        
        # Apply Well Dropout: Zero out wells where mask is True
        wells_input = cond.clone()
        mask_drop_wells_reshaped = mask_drop_wells.view(b, 1, 1, 1)
        wells_input = torch.where(mask_drop_wells_reshaped, torch.zeros_like(wells_input), wells_input)
        
        # Predict noise
        noise_pred = self.model(x_noisy, t, wells_input, labels, mask_drop_labels)
        
        # Diffusion loss (predict noise) - THIS IS THE ONLY LOSS
        loss_noise = F.mse_loss(noise_pred, noise)
        
        return {
            'total': loss_noise,
            'noise': loss_noise
        }
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise"""
        return (
            self.sqrt_recip_alphas_cumprod[t][:, None, None, None] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None] * noise
        )
    
    @torch.no_grad()
    def p_sample(self, x, t, cond, labels, mask_drop=None):
        """Single denoising step"""
        b = x.shape[0]
        device = x.device
        
        # Predict noise
        noise_pred = self.model(x, t, cond, labels, mask_drop)
        
        # Get parameters for this timestep
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None, None]
        
        # Predict x_0
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, cond, labels, guidance_scale=1.0, ignore_wells=False):
        """Full denoising loop with Batched CFG (Optimized)"""
        device = next(self.parameters()).device
        b = cond.shape[0]
        
        shape = (b, 1, self.config.IMG_SIZE, self.config.IMG_SIZE)
        
        # Start from noise
        img = torch.randn(shape, device=device)
        
        # Handle Wells
        if ignore_wells:
            wells_input = torch.zeros_like(cond)
        else:
            wells_input = cond

        # Prepare CFG masks
        # We will concatenate [Conditional, Unconditional]
        mask_keep = torch.zeros((b,), dtype=torch.bool, device=device) 
        mask_drop = torch.ones((b,), dtype=torch.bool, device=device)
        
        # Pre-prepare batch inputs that don't change to save CPU cycles
        wells_in = torch.cat([wells_input, wells_input], dim=0)
        labels_in = torch.cat([labels, labels], dim=0)
        mask_in = torch.cat([mask_keep, mask_drop], dim=0)

        # Iterate
        # Disable tqdm here if you are running this inside another progress bar in your test script
        for i in range(self.config.TIMESTEPS - 1, -1, -1):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            # --- OPTIMIZATION: BATCHED INFERENCE ---
            if guidance_scale != 1.0:
                # 1. Duplicate inputs for batching
                img_in = torch.cat([img, img], dim=0)
                t_in = torch.cat([t, t], dim=0)
                
                # 2. Single Forward Pass (Runs Cond and Uncond in parallel on GPU)
                noise_out = self.model(img_in, t_in, wells_in, labels_in, mask_drop=mask_in)
                
                # 3. Split results
                noise_cond, noise_uncond = noise_out.chunk(2, dim=0)
                
                # 4. Combine
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                # Standard path if no guidance
                noise_pred = self.model(img, t, wells_input, labels, mask_drop=mask_keep)
            
            # --- STANDARD UPDATE STEP ---
            betas_t = self.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None, None]
            
            model_mean = sqrt_recip_alphas_t * (img - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(img)
                var_t = self.posterior_variance[t][:, None, None, None]
                img = model_mean + torch.sqrt(var_t) * noise
            else:
                img = model_mean
        
        return img, labels
    
    @torch.no_grad()
    def sample(self, cond, labels=None, n_samples=1, guidance_scale=1.0, ignore_wells=False):
        """
        Generate samples conditioned on well observations.
        
        Args:
            cond: Well conditioning (B, 2, H, W) or single (2, H, W)
            labels: Variogram parameters (B, 4)
            n_samples: Number of samples per conditioning
        """
        if cond.dim() == 3:
            cond = cond.unsqueeze(0)
        
        b = cond.shape[0]
        
        if labels is None:
             # Default to 0 (mean) if no labels provided, though this shouldn't happen in training
             labels = torch.zeros((b, 4), device=cond.device)

        if n_samples > 1:
            cond = cond.repeat(n_samples, 1, 1, 1)
            labels = labels.repeat(n_samples, 1)
        
        images, params = self.p_sample_loop(cond, labels, guidance_scale, ignore_wells)
        
        return images, params


# =============================================================================
# DATASET
# =============================================================================

class VariogramDiffusionDataset(Dataset):
    """
    Dataset for training conditional diffusion model.
    """
    def __init__(self, npy_path, config, augment=True):
        print(f"Loading data from {npy_path}...")
        raw_data = np.load(npy_path, allow_pickle=True)
        
        # Filter unique realizations
        self.data = [d for d in raw_data if d.get('sample_idx', 0) == 0]
        
        self.config = config
        self.augment = augment
        
        print(f"Dataset ready: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def normalize_params(self, range_val, aniso_val, azimuth_val):
        """
        Normalize parameters to [-1, 1] range.
        REMOVED: Nugget parameter normalization.
        """
        ranges = self.config.PARAM_RANGES
        
        r = 2 * (range_val - ranges['range'][0]) / (ranges['range'][1] - ranges['range'][0]) - 1
        a = 2 * (aniso_val - ranges['aniso'][0]) / (ranges['aniso'][1] - ranges['aniso'][0]) - 1
        
        # Azimuth: use sin/cos encoding for periodicity
        azi_rad = np.deg2rad(2 * azimuth_val)  # Double angle for 180° periodicity
        sin_azi = np.sin(azi_rad)
        cos_azi = np.cos(azi_rad)
        
        return np.array([r, a, sin_azi, cos_azi], dtype=np.float32)
    
    def create_well_conditioning(self, field, n_wells):
        """Create sparse well conditioning tensor"""
        h, w = field.shape
        
        # Random well locations
        well_i = np.random.randint(0, h, size=n_wells)
        well_j = np.random.randint(0, w, size=n_wells)
        
        # Create conditioning channels
        well_values = np.zeros((h, w), dtype=np.float32)
        well_mask = np.zeros((h, w), dtype=np.float32)
        
        for i, j in zip(well_i, well_j):
            well_values[i, j] = field[i, j]
            well_mask[i, j] = 1.0
        
        return np.stack([well_values, well_mask], axis=0)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get field and normalize
        field = item['simulation'].squeeze()
        field_norm = np.clip(field, -self.config.DATA_SCALE, self.config.DATA_SCALE) / self.config.DATA_SCALE
        
        # Get parameters (Nugget removed from extraction)
        params = self.normalize_params(
            item['range'],
            item['anisotropy_ratio'],
            # item['nugget'] - REMOVED
            item['azimuth']
        )
        
        # Random number of wells
        n_wells = np.random.randint(self.config.MIN_WELLS, self.config.MAX_WELLS + 1)
        
        # Create conditioning
        cond = self.create_well_conditioning(field_norm, n_wells)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                field_norm = np.flip(field_norm, axis=1).copy()
                cond = np.flip(cond, axis=2).copy()
                # Flip azimuth (sin changes sign for horizontal flip)
                params[2] = -params[2]  # sin_azi is index 2 now
            
            # Random vertical flip
            if np.random.rand() > 0.5:
                field_norm = np.flip(field_norm, axis=0).copy()
                cond = np.flip(cond, axis=1).copy()
                # Flip azimuth
                params[2] = -params[2]  # sin_azi is index 2 now
        
        # Add channel dimension to field
        field_norm = field_norm[None, :, :]
        
        return (
            torch.tensor(field_norm, dtype=torch.float32),
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32)
        )


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# =============================================================================
# TRAINING
# =============================================================================

def train(config, resume=True):
    print("="*70)
    print("CONDITIONAL DIFFUSION MODEL TRAINING (NO NUGGET, NO RECON LOSS)")
    print("="*70)
    
    # Dataset and dataloader
    dataset = VariogramDiffusionDataset(config.DATA_PATH, config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # Model & Optimizer setup
    model = ConditionalDiffusion(config).to(config.DEVICE)
    ema = EMA(model, decay=config.EMA_DECAY)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS * len(dataloader))
    
    # State variables
    start_epoch = 0
    best_loss = float('inf')
    
    # Setup Logging File
    log_file = os.path.join(config.SAVE_DIR, 'training_log.csv')
    
    # --- RESUME LOGIC ---
    if resume:
        checkpoint_path = find_latest_checkpoint(config.SAVE_DIR)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from: {checkpoint_path}")
            try:
                checkpoint = torch.load('diffusion_checkpointsNONUGGETCONDV32_NO_RECON/checkpoint_epoch200.pth', map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                ema.load_state_dict(checkpoint['ema_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('loss', float('inf'))
                print(f"Resuming at Epoch {start_epoch+1}")
            except Exception as e:
                print(f"Could not resume from checkpoint: {e}")
                print("Starting from scratch...")
        else:
            # Create new log file header if starting fresh
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Total_Loss', 'Noise_Loss', 'LR'])
    else:
        # Create new log file header if starting fresh
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Total_Loss', 'Noise_Loss', 'LR'])
    # --------------------

    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        epoch_losses = {'total': 0, 'noise': 0}
        
        # Customized Progress Bar
        pbar = tqdm(dataloader, desc=f"Ep {epoch+1}")
        
        for batch_idx, (fields, conds, params) in enumerate(pbar):
            fields = fields.to(config.DEVICE)
            conds = conds.to(config.DEVICE)
            params = params.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Calculate losses
            losses = model.p_losses(fields, conds, params)
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            ema.update()
            
            # Accumulate
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            
            # Update Progress Bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}"
            })
        
        # Calculate Averages
        n_batches = len(dataloader)
        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        current_lr = scheduler.get_last_lr()[0]
        
        # Print Clean Summary Table
        print(f" ------------------------------------------------")
        print(f" | Epoch {epoch+1:03d} | Loss: {avg['total']:.4f} | LR: {current_lr:.1e} |")
        print(f" ------------------------------------------------")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg['total'], avg['noise'], current_lr])

        # Save Checkpoints (Best & Regular)
        if avg['total'] < best_loss:
            best_loss = avg['total']
            save_checkpoint(model, optimizer, scheduler, ema, config, epoch, best_loss, 'best')
            
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, ema, config, epoch, avg['total'], f'epoch{epoch+1}')
        
        # Generate samples
        if (epoch + 1) % 50 == 0:
            generate_samples(model, dataset, config, epoch, ema)
    
    return model, ema

def save_checkpoint(model, optimizer, scheduler, ema, config, epoch, loss, suffix):
    ema.apply_shadow()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'config': config.__dict__,
        'loss': loss
    }, f"{config.SAVE_DIR}/checkpoint_{suffix}.pth")
    ema.restore()


def generate_samples(model, dataset, config, epoch, ema=None):
    """Generate and visualize samples"""
    model.eval()
    
    if ema is not None:
        ema.apply_shadow()
    
    # Get a sample from dataset
    field, cond, params = dataset[0]
    cond = cond.unsqueeze(0).to(config.DEVICE)
    params = params.unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        generated, _ = model.sample(cond, params, n_samples=4, guidance_scale=2.0)
    
    if ema is not None:
        ema.restore()
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original field
    axes[0, 0].imshow(field[0].numpy() * config.DATA_SCALE, cmap='jet', vmin=-3, vmax=3)
    axes[0, 0].set_title('Original Field')
    
    # Well locations
    well_mask = cond[0, 1].cpu().numpy()
    well_values = cond[0, 0].cpu().numpy()
    axes[0, 1].imshow(well_values, cmap='jet', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'Well Observations (n={int(well_mask.sum())})')
    
    # Generated samples
    for i in range(4):
        ax = axes[(i+2)//3, (i+2)%3]
        gen = generated[i, 0].cpu().numpy() * config.DATA_SCALE
        ax.imshow(gen, cmap='jet', vmin=-3, vmax=3)
        ax.set_title(f'Generated {i+1}')
    
    plt.suptitle(f'Epoch {epoch+1}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{config.SAVE_DIR}/samples_epoch{epoch+1}.png", dpi=150)
    plt.close()
    
    model.train()


# =============================================================================
# INFERENCE
# =============================================================================

class DiffusionGeneration:
    """
    Generation class for variogram parameter conditioning.
    """
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate config
        self.config = Config()
        for k, v in checkpoint['config'].items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        # Load model
        self.model = ConditionalDiffusion(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
    
    def create_conditioning(self, well_locations, well_values, grid_size=128):
        """
        Create conditioning tensor from well data.
        
        Args:
            well_locations: (N, 2) array of [i, j] coordinates
            well_values: (N,) array of values at wells
            grid_size: Size of the grid
        
        Returns:
            cond: (1, 2, H, W) conditioning tensor
        """
        well_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        mask = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        for (i, j), v in zip(well_locations, well_values):
            well_map[i, j] = v / self.config.DATA_SCALE  # Normalize
            mask[i, j] = 1.0
        
        cond = np.stack([well_map, mask], axis=0)
        return torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def normalize_params(self, range_val, aniso_val, azimuth_val):
        """
        Normalize parameters to [-1, 1] range.
        """
        ranges = self.config.PARAM_RANGES
        
        r = 2 * (range_val - ranges['range'][0]) / (ranges['range'][1] - ranges['range'][0]) - 1
        a = 2 * (aniso_val - ranges['aniso'][0]) / (ranges['aniso'][1] - ranges['aniso'][0]) - 1
        
        # Azimuth: use sin/cos encoding
        azi_rad = np.deg2rad(2 * azimuth_val)
        sin_azi = np.sin(azi_rad)
        cos_azi = np.cos(azi_rad)
        
        return torch.tensor([r, a, sin_azi, cos_azi], dtype=torch.float32).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def generate(self, well_locations, well_values, params, n_samples=1, guidance_scale=1.0):
        """
        Generate fields conditioned on wells and parameters.
        
        Args:
            well_locations: (N, 2) array of [i, j] coordinates
            well_values: (N,) array of values at wells
            params: Dict with 'range', 'aniso', 'azimuth'
            n_samples: Number of samples to generate
        
        Returns:
            realizations: Generated fields
        """
        # Create conditioning
        cond = self.create_conditioning(well_locations, well_values)
        labels = self.normalize_params(params['range'], params['aniso'], params['azimuth'])
        
        # Generate samples
        all_realizations = []
        
        batch_size = min(n_samples, 10)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Generating samples"):
            current_batch = min(batch_size, n_samples - i * batch_size)
            
            # Sample
            realizations, _ = self.model.sample(cond, labels, n_samples=current_batch, guidance_scale=guidance_scale)
            
            all_realizations.append(realizations.cpu())
        
        all_realizations = torch.cat(all_realizations, dim=0).numpy()
        
        return all_realizations * self.config.DATA_SCALE


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint for testing')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Pass the resume flag to the train function
        model, ema = train(config, resume=True)
    
    else:
        # Test inference
        if args.checkpoint is None:
            # Try to find the best model if none specified
            args.checkpoint = f"{config.SAVE_DIR}/checkpoint_epoch200.pth"
            if not os.path.exists(args.checkpoint):
                # Fallback to latest epoch if best doesn't exist
                args.checkpoint = find_latest_checkpoint(config.SAVE_DIR)
        
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"No checkpoint found at {args.checkpoint}")

        generator = DiffusionGeneration(args.checkpoint)
        
        # Example generation
        print("Generating sample...")
        
        # Dummy wells
        well_locs = np.array([[64, 64], [32, 32]])
        well_vals = np.array([1.0, -1.0])
        
        # Dummy params
        params = {
            'range': 300,
            'aniso': 2.0,
            'azimuth': 45
        }
        
        realizations = generator.generate(well_locs, well_vals, params, n_samples=4, guidance_scale=2.0)
        
        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            axes[i].imshow(realizations[i, 0], cmap='jet')
            axes[i].set_title(f'Sample {i+1}')
        plt.show()