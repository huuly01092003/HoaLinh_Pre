"""
Complete Time-Series Training Script
Run: python train_timeseries.py --env development --epochs 20
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from datetime import datetime

# Force immediate output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("\n[TRAIN] Starting imports...", flush=True)

from config import get_config
from model import create_timeseries_model

print("[TRAIN] All imports successful!", flush=True)


def setup_logging(config):
    """Setup logging"""
    log_file = config.LOGS_DIR / f'training_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def load_checkpoint_data(config):
    """Load data from checkpoint saved by main.py"""
    checkpoint_path = config.MODEL_DIR / 'timeseries_checkpoint.pth'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please run 'python main.py --env development' first!"
        )
    
    print(f"\n[LOAD] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    
    print(f"[LOAD] ✅ Checkpoint loaded:")
    print(f"  - Products: {checkpoint['num_products']}")
    print(f"  - Encoders: {len(checkpoint['encoders'])}")
    
    return checkpoint


class TimeAwareLoss(nn.Module):
    """
    Time-aware loss that weights recent predictions more heavily
    """
    def __init__(self, alpha_product=1.0, alpha_qty=0.5, alpha_revenue=1.5, alpha_discount=0.3):
        super().__init__()
        self.alpha_product = alpha_product
        self.alpha_qty = alpha_qty
        self.alpha_revenue = alpha_revenue
        self.alpha_discount = alpha_discount
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.SmoothL1Loss()
    
    def forward(self, outputs, targets):
        """
        outputs: dict with keys 'product', 'quantity', 'revenue', 'discount'
        targets: dict with same keys
        """
        loss_product = self.criterion_cls(outputs['product'], targets['product'])
        loss_qty = self.criterion_reg(outputs['quantity'], targets['quantity'])
        loss_revenue = self.criterion_reg(outputs['revenue'], targets['revenue'])
        loss_discount = self.criterion_reg(outputs['discount'], targets['discount'])
        
        # Weighted multi-task loss
        total_loss = (
            self.alpha_product * loss_product +
            self.alpha_qty * loss_qty +
            self.alpha_revenue * loss_revenue +
            self.alpha_discount * loss_discount
        )
        
        return total_loss, {
            'product': loss_product.item(),
            'quantity': loss_qty.item(),
            'revenue': loss_revenue.item(),
            'discount': loss_discount.item()
        }


class TimeSeriesTrainer:
    """Training orchestrator for time-series model"""
    
    def __init__(self, model, train_loader, val_loader, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        
        # Loss function
        self.criterion = TimeAwareLoss()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training", ncols=100)
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward
            self.optimizer.zero_grad()
            
            outputs = self.model(
                batch['product_ids'],
                batch['qty'],
                batch['revenue'],
                batch['discount'],
                batch['customer_features'],
                batch['target_time_features']
            )
            
            # Loss
            targets = {
                'product': batch['target_product'],
                'quantity': batch['target_qty'],
                'revenue': batch['target_revenue'],
                'discount': batch['target_discount']
            }
            
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs['product'].max(1)
            total_correct += (predicted == targets['product']).sum().item()
            total_samples += targets['product'].size(0)
            
            # Update progress
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct/total_samples:.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", ncols=100):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                batch['product_ids'],
                batch['qty'],
                batch['revenue'],
                batch['discount'],
                batch['customer_features'],
                batch['target_time_features']
            )
            
            targets = {
                'product': batch['target_product'],
                'quantity': batch['target_qty'],
                'revenue': batch['target_revenue'],
                'discount': batch['target_discount']
            }
            
            loss, _ = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs['product'].max(1)
            total_correct += (predicted == targets['product']).sum().item()
            total_samples += targets['product'].size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, epochs):
        """Full training loop"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING TIME-SERIES TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
        self.logger.info("="*80 + "\n")
        
        for epoch in range(epochs):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch+1}/{epochs}")
            self.logger.info("="*80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            self.logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Save best
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"✅ Best model saved! (improved by {improvement:.4f})")
            else:
                self.patience_counter += 1
                self.logger.info(f"⚠️ No improvement ({self.patience_counter}/{self.config.PATIENCE})")
                
                if self.patience_counter >= self.config.PATIENCE:
                    self.logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # Memory cleanup
            gc.collect()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("✅ TRAINING COMPLETE!")
        self.logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        self.logger.info("="*80 + "\n")
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = self.config.MODEL_DIR / 'best_timeseries_model.pth'
        else:
            path = self.config.MODEL_DIR / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
        self.logger.info(f"💾 Saved: {path.name}")


def main(args):
    """Main training pipeline"""
    
    print("\n[TRAIN] Starting training pipeline...", flush=True)
    
    # Config
    config = get_config(args.env)
    logger = setup_logging(config)
    
    logger.info("\n" + "="*80)
    logger.info("TIME-SERIES TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("="*80 + "\n")
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint_data(config)
    num_products = checkpoint_data['num_products']
    
    # Load dataloaders (they should be saved from main.py)
    # For now, we'll need to recreate them from sequences
    logger.info("\n[LOAD] Loading processed sequences...")
    
    # Load sequences from processed directory
    from data_manager import DataManager
    data_manager = DataManager(config.PROCESSED_DATA_PATH, use_compression=True)
    
    try:
        train_sequences, val_sequences, test_sequences = data_manager.load_sequences('latest')
        product_to_id, encoders = data_manager.load_mappings('latest')
        logger.info("✅ Loaded from cache!")
    except Exception as e:
        logger.error(f"Failed to load sequences: {e}")
        logger.error("Please run 'python main.py --env development' first!")
        return
    
    # Create datasets and loaders
    logger.info("\n[DATASET] Creating dataloaders...")
    from dataset import create_dataloaders
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_sequences, val_sequences, test_sequences,
        product_to_id, config
    )
    
    # Create model
    logger.info("\n[MODEL] Creating time-series model...")
    model = create_timeseries_model(config, num_products)
    
    # Train
    trainer = TimeSeriesTrainer(model, train_loader, val_loader, config, logger)
    history = trainer.train(args.epochs)
    
    # Final save
    logger.info("\n[SAVE] Saving final model...")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'product_to_id': product_to_id,
        'num_products': num_products,
        'encoders': encoders,
        'history': history,
        'config': config.to_dict()
    }
    
    final_path = config.MODEL_DIR / 'final_timeseries_model.pth'
    torch.save(final_checkpoint, final_path)
    logger.info(f"✅ Final model saved: {final_path}")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ALL DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Time-Series Model')
    parser.add_argument('--env', type=str, default='development',
                       choices=['development', 'production', 'fast'])
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n[TRAIN] ❌ FATAL ERROR: {e}", flush=True)
        logging.error(f"Training failed: {e}", exc_info=True)
        raise