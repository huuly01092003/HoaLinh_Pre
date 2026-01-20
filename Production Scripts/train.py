"""
Optimized Training Module for Large Datasets
Handles 2.4M+ rows efficiently with gradient accumulation and mixed precision
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import logging
from typing import Dict, Tuple
import gc

logger = logging.getLogger(__name__)


class OptimizedTrainer:
    """Memory-efficient training orchestrator for large datasets"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        save_dir: Path = None,
        accumulation_steps: int = 4  # Gradient accumulation
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = save_dir or config.MODEL_DIR
        self.accumulation_steps = accumulation_steps
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        
        # Loss functions
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.SmoothL1Loss()
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_mae': [], 'val_rmse': [],
            'lr': []
        }
        
        logger.info(f"✅ Trainer initialized with gradient accumulation: {accumulation_steps}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        product_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training", ncols=100)
        
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                batch['product_ids'],
                batch['qty'],
                batch['revenue'],
                batch['discount'],
                batch['week_change'],
                batch['month_change'],
                batch['quarter_change'],
                batch['year_change'],
                batch['customer_features']
            )
            
            # Calculate losses
            loss_product = self.criterion_cls(outputs['product'], batch['target_product'])
            loss_qty = self.criterion_reg(outputs['quantity'], batch['target_qty'])
            loss_revenue = self.criterion_reg(outputs['revenue'], batch['target_revenue'])
            loss_discount = self.criterion_reg(outputs['discount'], batch['target_discount'])
            
            # Weighted multi-task loss
            loss = (
                1.0 * loss_product +
                0.5 * loss_qty +
                1.5 * loss_revenue +
                0.3 * loss_discount
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation: only step every N batches
            if (i + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRADIENT_CLIP
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs['product'].max(1)
            product_correct += (predicted == batch['target_product']).sum().item()
            total_samples += batch['target_product'].size(0)
            
            # Update progress bar
            current_acc = product_correct / total_samples
            pbar.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'acc': f"{current_acc:.4f}"
            })
            
            # Clean up
            if i % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Final step if needed
        if len(self.train_loader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRADIENT_CLIP
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = product_correct / total_samples
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader = None) -> Tuple[float, float, float, float]:
        """Validate model efficiently"""
        if dataloader is None:
            dataloader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        product_correct = 0
        total_samples = 0
        
        # Use lists for memory efficiency
        revenue_preds = []
        revenue_targets = []
        
        pbar = tqdm(dataloader, desc="Validating", ncols=100)
        
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                batch['product_ids'],
                batch['qty'],
                batch['revenue'],
                batch['discount'],
                batch['week_change'],
                batch['month_change'],
                batch['quarter_change'],
                batch['year_change'],
                batch['customer_features']
            )
            
            # Losses
            loss_product = self.criterion_cls(outputs['product'], batch['target_product'])
            loss_qty = self.criterion_reg(outputs['quantity'], batch['target_qty'])
            loss_revenue = self.criterion_reg(outputs['revenue'], batch['target_revenue'])
            loss_discount = self.criterion_reg(outputs['discount'], batch['target_discount'])
            
            loss = (
                1.0 * loss_product +
                0.5 * loss_qty +
                1.5 * loss_revenue +
                0.3 * loss_discount
            )
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = outputs['product'].max(1)
            product_correct += (predicted == batch['target_product']).sum().item()
            total_samples += batch['target_product'].size(0)
            
            # Collect predictions (move to CPU immediately)
            revenue_preds.extend(outputs['revenue'].cpu().squeeze().tolist())
            revenue_targets.extend(batch['target_revenue'].cpu().squeeze().tolist())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = product_correct / total_samples
        mae = mean_absolute_error(revenue_targets, revenue_preds)
        rmse = np.sqrt(mean_squared_error(revenue_targets, revenue_preds))
        
        return avg_loss, accuracy, mae, rmse
    
    def train(self, epochs: int = None):
        """Full training loop with memory management"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Gradient accumulation steps: {self.accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.BATCH_SIZE * self.accumulation_steps}")
        logger.info(f"{'='*80}\n")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info("="*80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_mae, val_rmse = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            self.history['lr'].append(current_lr)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            logger.info(f"Revenue MAE: {val_mae:,.0f} | RMSE: {val_rmse:,.0f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"✅ Best model saved! (improved by {improvement:.4f})")
            else:
                self.patience_counter += 1
                logger.info(f"⚠️  No improvement ({self.patience_counter}/{self.config.PATIENCE})")
                
                if self.patience_counter >= self.config.PATIENCE:
                    logger.info(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("\n" + "="*80)
        logger.info("✅ Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("="*80 + "\n")
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.to_dict()
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
        logger.info(f"💾 Checkpoint saved: {path.name}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)
        logger.info(f"✅ Checkpoint loaded from {path}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, float], Dict, Dict]:
    """Evaluate model on test set efficiently"""
    model.eval()
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    
    product_correct = 0
    total_samples = 0
    
    all_preds = {
        'product': [], 'quantity': [], 'revenue': [], 'discount': []
    }
    all_targets = {
        'product': [], 'quantity': [], 'revenue': [], 'discount': []
    }
    
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*80)
    
    for batch in tqdm(test_loader, desc="Testing", ncols=100):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(
            batch['product_ids'],
            batch['qty'],
            batch['revenue'],
            batch['discount'],
            batch['week_change'],
            batch['month_change'],
            batch['quarter_change'],
            batch['year_change'],
            batch['customer_features']
        )
        
        # Collect predictions and targets (move to CPU immediately)
        _, predicted = outputs['product'].max(1)
        all_preds['product'].extend(predicted.cpu().tolist())
        all_targets['product'].extend(batch['target_product'].cpu().tolist())
        
        all_preds['quantity'].extend(outputs['quantity'].cpu().squeeze().tolist())
        all_targets['quantity'].extend(batch['target_qty'].cpu().squeeze().tolist())
        
        all_preds['revenue'].extend(outputs['revenue'].cpu().squeeze().tolist())
        all_targets['revenue'].extend(batch['target_revenue'].cpu().squeeze().tolist())
        
        all_preds['discount'].extend(outputs['discount'].cpu().squeeze().tolist())
        all_targets['discount'].extend(batch['target_discount'].cpu().squeeze().tolist())
        
        product_correct += (predicted == batch['target_product']).sum().item()
        total_samples += batch['target_product'].size(0)
    
    # Calculate metrics
    metrics = {
        'product_accuracy': product_correct / total_samples,
        'revenue_mae': mean_absolute_error(all_targets['revenue'], all_preds['revenue']),
        'revenue_rmse': np.sqrt(mean_squared_error(all_targets['revenue'], all_preds['revenue'])),
        'quantity_mae': mean_absolute_error(all_targets['quantity'], all_preds['quantity']),
        'quantity_rmse': np.sqrt(mean_squared_error(all_targets['quantity'], all_preds['quantity'])),
        'discount_mae': mean_absolute_error(all_targets['discount'], all_preds['discount'])
    }
    
    logger.info("\n📊 TEST RESULTS:")
    logger.info(f"  Product Accuracy: {metrics['product_accuracy']*100:.2f}%")
    logger.info(f"  Revenue MAE: {metrics['revenue_mae']:,.0f} VND")
    logger.info(f"  Revenue RMSE: {metrics['revenue_rmse']:,.0f} VND")
    logger.info(f"  Quantity MAE: {metrics['quantity_mae']:.2f}")
    logger.info(f"  Quantity RMSE: {metrics['quantity_rmse']:.2f}")
    logger.info(f"  Discount MAE: {metrics['discount_mae']:.4f}")
    logger.info("="*80 + "\n")
    
    return metrics, all_preds, all_targets