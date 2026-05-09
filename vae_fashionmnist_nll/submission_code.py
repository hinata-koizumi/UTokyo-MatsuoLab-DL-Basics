"""
VAE FashionMNIST Submission Code
Single-file implementation for reproducing LB score 201.556

DATA USAGE:
  - Training uses ONLY x_train.npy (no external data sources)
  - x_test.npy is used STRICTLY for inference to generate submission CSV
  - No oracle, test-time optimization, or x_test information is used for:
    * Hyperparameter tuning
    * Checkpoint selection
    * Model architecture decisions

Usage:
  # Train (optional, for reproducibility)
  python submission_code.py --mode train --seed 0 --epochs 22
  
  # Generate ensemble submission (reproduces final result)
  python submission_code.py --mode ensemble --data_dir data/input --out_csv submission.csv
  
  # Predict single model
  python submission_code.py --mode predict --ckpt checkpoints/seed0/model_epoch16.pth --out_csv pred.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================================
# 1. IMPORTS + UTILITIES
# ============================================================================

def set_seed(seed):
    """Fix random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Auto-detect device (CUDA > MPS > CPU for compatibility)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def sigmoid(x):
    """Numerically stable sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))

def clip_probs(probs, eps=1e-7):
    """Clip probabilities to [eps, 1-eps]"""
    return np.clip(probs, eps, 1.0 - eps)

def calculate_nll(probs, targets):
    """Calculate NLL (BCE) per image - local verification only"""
    probs_clipped = np.clip(probs, 1e-10, 1.0 - 1e-10)
    bce = -(targets * np.log(probs_clipped) + (1 - targets) * np.log(1 - probs_clipped))
    return bce.sum(axis=1).mean()

# ============================================================================
# 2. CONFIG (CONSTANTS)
# ============================================================================

# Final submission configuration (LB 201.556)
FINAL_CONFIG = {
    'eps': 1e-7,
    'batch_size': 128,
    'epochs': 22,
    'z_dim': 64,
    'base_channels': 128,
    'w_bg': 3.0,
    'beta': 1.0,
    'use_skip': True,
    'lr': 1e-3,
    'data_seed': 42,  # Fixed for train/valid split
}

# Final 20-model ensemble (seeds × epochs)
# Fixed checkpoints (same epochs for all seeds); no x_test used for any selection.
FINAL_ENSEMBLE_CKPTS = [
    # Seed 0
    'checkpoints/phase5_ckpt_ens/seed0/model_epoch16.pth',
    'checkpoints/phase5_ckpt_ens/seed0/model_epoch18.pth',
    'checkpoints/phase5_ckpt_ens/seed0/model_epoch20.pth',
    'checkpoints/phase5_ckpt_ens/seed0/model_epoch22.pth',
    # Seed 1
    'checkpoints/phase5_ckpt_ens/seed1/model_epoch16.pth',
    'checkpoints/phase5_ckpt_ens/seed1/model_epoch18.pth',
    'checkpoints/phase5_ckpt_ens/seed1/model_epoch20.pth',
    'checkpoints/phase5_ckpt_ens/seed1/model_epoch22.pth',
    # Seed 2
    'checkpoints/phase5_ckpt_ens/seed2/model_epoch16.pth',
    'checkpoints/phase5_ckpt_ens/seed2/model_epoch18.pth',
    'checkpoints/phase5_ckpt_ens/seed2/model_epoch20.pth',
    'checkpoints/phase5_ckpt_ens/seed2/model_epoch22.pth',
    # Seed 3
    'checkpoints/phase5_ckpt_ens/seed3/model_epoch16.pth',
    'checkpoints/phase5_ckpt_ens/seed3/model_epoch18.pth',
    'checkpoints/phase5_ckpt_ens/seed3/model_epoch20.pth',
    'checkpoints/phase5_ckpt_ens/seed3/model_epoch22.pth',
    # Seed 4
    'checkpoints/phase5_ckpt_ens/seed4/model_epoch16.pth',
    'checkpoints/phase5_ckpt_ens/seed4/model_epoch18.pth',
    'checkpoints/phase5_ckpt_ens/seed4/model_epoch20.pth',
    'checkpoints/phase5_ckpt_ens/seed4/model_epoch22.pth',
]

# ============================================================================
# 3. DATASET / DATALOADER
# ============================================================================

class FashionMNISTDataset(Dataset):
    """Load FashionMNIST from .npy files"""
    def __init__(self, data_path):
        # Load uint8 data
        data = np.load(data_path)
        # Normalize to [0, 1]
        self.data = (data.astype(np.float32) / 255.0)
        # Reshape to (N, 1, 28, 28)
        if self.data.ndim == 3:
            self.data = self.data[:, np.newaxis, :, :]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        return x, 0  # Return dummy label for compatibility

def load_data(data_dir, batch_size=128, seed=42):
    """Load train/valid/test dataloaders"""
    train_path = os.path.join(data_dir, 'x_train.npy')
    test_path = os.path.join(data_dir, 'x_test.npy')
    
    # Load full training set
    full_train_dataset = FashionMNISTDataset(train_path)
    
    # Split train/valid (fixed seed for consistency)
    generator = torch.Generator().manual_seed(seed)
    n_train = int(0.9 * len(full_train_dataset))
    n_valid = len(full_train_dataset) - n_train
    train_dataset, valid_dataset = random_split(
        full_train_dataset, [n_train, n_valid], generator=generator
    )
    
    # Load test set
    test_dataset = FashionMNISTDataset(test_path)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

# ============================================================================
# 4. MODEL DEFINITION
# ============================================================================

class BaseVAE(nn.Module):
    """VAE with skip connections"""
    def __init__(self, z_dim=64, base_channels=128, use_skip=True):
        super(BaseVAE, self).__init__()
        self.z_dim = z_dim
        self.use_skip = use_skip
        self.base_channels = base_channels
        c = base_channels

        # Encoder: 1 -> c (14x14) -> 2c (7x7)
        self.enc_conv1 = nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1)
        self.enc_relu1 = nn.ReLU()
        self.enc_conv2 = nn.Conv2d(c, c*2, kernel_size=3, stride=2, padding=1)
        self.enc_relu2 = nn.ReLU()
        
        # Bottleneck
        flat_dim = (c * 2) * 7 * 7
        self.fc_mu = nn.Linear(flat_dim, z_dim)
        self.fc_logvar = nn.Linear(flat_dim, z_dim)

        # Decoder
        self.decoder_input = nn.Linear(z_dim, flat_dim)
        self.dec_unflatten = nn.Unflatten(1, (c * 2, 7, 7))
        self.dec_convt1 = nn.ConvTranspose2d(c * 2, c, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_relu1 = nn.ReLU()
        self.dec_convt2 = nn.ConvTranspose2d(c, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Skip connections
        if use_skip:
            self.skip_conv_7 = nn.Conv2d(c * 2, c * 2, kernel_size=1)
            self.skip_conv_14 = nn.Conv2d(c, c, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        f14 = self.enc_relu1(self.enc_conv1(x))
        f7 = self.enc_relu2(self.enc_conv2(f14))
        
        # Bottleneck
        h = f7.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z_dec = self.decoder_input(z)
        d7 = self.dec_unflatten(z_dec)
        
        if self.use_skip:
            d7 = d7 + self.skip_conv_7(f7)
        
        d14 = self.dec_relu1(self.dec_convt1(d7))
        
        if self.use_skip:
            d14 = d14 + self.skip_conv_14(f14)
        
        logits = self.dec_convt2(d14)
        x_hat = torch.sigmoid(logits)
        
        return x_hat, mu, logvar
    
    def get_logits(self, x):
        """Get logits without sigmoid (for ensemble)"""
        # Encode
        f14 = self.enc_relu1(self.enc_conv1(x))
        f7 = self.enc_relu2(self.enc_conv2(f14))
        
        # Bottleneck
        h = f7.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z_dec = self.decoder_input(z)
        d7 = self.dec_unflatten(z_dec)
        
        if self.use_skip:
            d7 = d7 + self.skip_conv_7(f7)
        
        d14 = self.dec_relu1(self.dec_convt1(d7))
        
        if self.use_skip:
            d14 = d14 + self.skip_conv_14(f14)
        
        logits = self.dec_convt2(d14)
        return logits

# ============================================================================
# 5. TRAINING
# ============================================================================

def vae_loss(x, x_hat, mu, logvar, beta=1.0, w_bg=3.0):
    """
    VAE loss with background weighting
    Background is defined as pixels with x < 0.5
    """
    # Reconstruction loss (BCE)
    bce = F.binary_cross_entropy(x_hat, x, reduction='none')
    
    # Background weighting
    bg_mask = (x < 0.5).float()
    weighted_bce = bce * (1.0 + (w_bg - 1.0) * bg_mask)
    recon_loss = weighted_bce.sum(dim=[1, 2, 3]).mean()
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_one_epoch(model, train_loader, optimizer, device, beta=1.0, w_bg=3.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        
        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta, w_bg)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
    }

def evaluate(model, valid_loader, device):
    """Evaluate on validation set"""
    model.eval()
    total_nll = 0
    
    with torch.no_grad():
        for x, _ in valid_loader:
            x = x.to(device)
            x_hat, _, _ = model(x)
            
            # Calculate NLL
            bce = F.binary_cross_entropy(x_hat, x, reduction='none')
            nll = bce.sum(dim=[1, 2, 3]).mean()
            total_nll += nll.item()
    
    return total_nll / len(valid_loader)

def train(data_dir, out_dir, seed=0, epochs=22, batch_size=128, lr=1e-3,
          z_dim=64, base_channels=128, use_skip=True, w_bg=3.0, beta=1.0):
    """Full training loop"""
    # Setup
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    save_dir = os.path.join(out_dir, 'phase5_ckpt_ens', f'seed{seed}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    train_loader, valid_loader, _ = load_data(data_dir, batch_size, seed=42)
    
    # Create model
    model = BaseVAE(z_dim=z_dim, base_channels=base_channels, use_skip=use_skip).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"Starting training for Seed {seed}...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, beta, w_bg)
        val_nll = evaluate(model, valid_loader, device)
        
        print(f"Epoch {epoch}/{epochs} | Loss: {train_metrics['loss']:.1f} | Val NLL: {val_nll:.2f}")
        
        # Save checkpoints at specific epochs
        if epoch in [12, 14, 16, 18, 20, 22]:
            ckpt_path = os.path.join(save_dir, f'model_epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}")
    
    print(f"Finished training Seed {seed}")

# ============================================================================
# 6. PREDICTION
# ============================================================================

def predict_single(ckpt_path, data_dir, out_csv=None, device=None,
                   z_dim=64, base_channels=128, use_skip=True, eps=1e-7):
    """Predict with single model and save CSV"""
    if device is None:
        device = get_device()
    
    # Load model
    model = BaseVAE(z_dim=z_dim, base_channels=base_channels, use_skip=use_skip).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load test data
    test_path = os.path.join(data_dir, 'x_test.npy')
    test_dataset = FashionMNISTDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Predict
    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat, _, _ = model(x)
            probs = x_hat.view(x_hat.size(0), -1).cpu().numpy()
            all_probs.append(probs)
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_probs = clip_probs(all_probs, eps)
    
    # Save CSV if requested
    if out_csv:
        df = pd.DataFrame(all_probs)
        df.to_csv(out_csv, header=False, index=False, float_format='%.8f')
        print(f"Saved predictions to {out_csv}")
    
    return all_probs

def predict_logits(ckpt_path, data_dir, device=None,
                   z_dim=64, base_channels=128, use_skip=True):
    """Predict logits (for ensemble)"""
    if device is None:
        device = get_device()
    
    # Load model
    model = BaseVAE(z_dim=z_dim, base_channels=base_channels, use_skip=use_skip).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load test data
    test_path = os.path.join(data_dir, 'x_test.npy')
    test_dataset = FashionMNISTDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Predict logits
    all_logits = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model.get_logits(x)
            logits = logits.view(logits.size(0), -1).cpu().numpy()
            all_logits.append(logits)
    
    all_logits = np.concatenate(all_logits, axis=0)
    return all_logits

# ============================================================================
# 7. ENSEMBLE
# ============================================================================

def ensemble(ckpt_paths, data_dir, out_csv, eps=1e-7,
             z_dim=64, base_channels=128, use_skip=True):
    """Ensemble multiple models via logit averaging"""
    device = get_device()
    print(f"Using device: {device}")
    print(f"Ensembling {len(ckpt_paths)} models...")
    
    # Collect logits from all models
    logit_list = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"[{i+1}/{len(ckpt_paths)}] Loading {ckpt_path}...")
        logits = predict_logits(ckpt_path, data_dir, device, z_dim, base_channels, use_skip)
        logit_list.append(logits)
    
    # Average logits
    print("Averaging logits...")
    logits_stack = np.stack(logit_list, axis=0)
    logits_mean = np.mean(logits_stack, axis=0)
    
    # Sigmoid and clip
    probs = sigmoid(logits_mean)
    probs_clipped = clip_probs(probs, eps)
    
    # Save CSV
    df = pd.DataFrame(probs_clipped)
    df.to_csv(out_csv, header=False, index=False, float_format='%.8f')
    print(f"Saved ensemble to {out_csv}")
    print(f"Shape: {probs_clipped.shape}, Range: [{probs_clipped.min():.8f}, {probs_clipped.max():.8f}]")

# ============================================================================
# 8. MAIN / ARGPARSE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='VAE FashionMNIST Submission Code')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'predict', 'ensemble'],
                        help='Mode: train, predict, or ensemble')
    
    # Common args
    parser.add_argument('--data_dir', type=str, default='data/input',
                        help='Directory containing x_train.npy and x_test.npy')
    parser.add_argument('--out_csv', type=str, default='submission.csv',
                        help='Output CSV path')
    
    # Training args
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for training')
    parser.add_argument('--epochs', type=int, default=22,
                        help='Number of training epochs')
    parser.add_argument('--out_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    
    # Prediction args
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path for single model prediction')
    
    # Ensemble args
    parser.add_argument('--ckpts', type=str, nargs='+', default=None,
                        help='List of checkpoint paths for ensemble (overrides default)')
    parser.add_argument('--eps', type=float, default=1e-7,
                        help='Clipping epsilon')
    
    # Model args (fixed - must match trained checkpoints)
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Latent dimension')
    parser.add_argument('--base_channels', type=int, default=128,
                        help='Base channels')
    # use_skip is fixed to True to match the provided checkpoints
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=FINAL_CONFIG['batch_size'],
            lr=FINAL_CONFIG['lr'],
            z_dim=args.z_dim,
            base_channels=args.base_channels,
            use_skip=True,  # Fixed - all checkpoints use skip connections
            w_bg=FINAL_CONFIG['w_bg'],
            beta=FINAL_CONFIG['beta'],
        )
    
    elif args.mode == 'predict':
        if args.ckpt is None:
            raise ValueError("--ckpt is required for predict mode")
        predict_single(
            ckpt_path=args.ckpt,
            data_dir=args.data_dir,
            out_csv=args.out_csv,
            z_dim=args.z_dim,
            base_channels=args.base_channels,
            use_skip=True,  # Fixed - all checkpoints use skip connections
            eps=args.eps,
        )
    
    elif args.mode == 'ensemble':
        # Use provided ckpts or default final ensemble
        ckpt_paths = args.ckpts if args.ckpts else FINAL_ENSEMBLE_CKPTS
        ensemble(
            ckpt_paths=ckpt_paths,
            data_dir=args.data_dir,
            out_csv=args.out_csv,
            eps=args.eps,
            z_dim=args.z_dim,
            base_channels=args.base_channels,
            use_skip=True,  # Fixed - all checkpoints use skip connections
        )

if __name__ == '__main__':
    main()
