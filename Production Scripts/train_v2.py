"""
Complete Training Pipeline V2 - SIMPLIFIED
Focus: Products + Quantities (NO MONEY)
Run: python train_v2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import json
from datetime import datetime
import pickle

from config_v2 import SystemConfig
from data_loader_v2 import DataLoaderV2
from sequence_builder_v2 import SequenceBuilderV2, VocabularyBuilder, NormalizationStats
from dataset_v2 import create_dataloaders_v2
from tft_model_v2 import create_tft_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiTaskLossV2(nn.Module):
    """
    Multi-task loss - SIMPLIFIED (NO MONEY)
    Focus: Products > Quantities > Num SKUs > Days
    """
    
    def __init__(self):
        super().__init__()
        
        # SIMPLIFIED weights - FOCUS ON PRODUCTS
        self.w_products = 5.0      # HIGHEST: Product prediction là quan trọng nhất
        self.w_total_qty = 2.0     # MEDIUM: Quantity prediction
        self.w_num_skus = 1.0      # LOW: Num SKUs prediction
        self.w_days = 0.5          # LOWEST: Days prediction
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss()
    
    def forward(self, predictions, targets):
        """Calculate multi-task loss - SIMPLIFIED"""
        
        # 1. Product prediction (MOST IMPORTANT)
        target_first_product = targets['target_products'][:, 0]
        loss_products = self.ce_loss(predictions['products'], target_first_product)
        
        # 2. Total quantity prediction
        loss_total_qty = self.huber_loss(
            predictions['total_qty'],
            targets['target_total_qty']
        )
        
        # 3. Number of SKUs
        loss_num_skus = self.mse_loss(
            predictions['num_skus'],
            targets['target_num_skus']
        )
        
        # 4. Days until next order
        loss_days = self.mse_loss(
            predictions['days_until_next'],
            targets['days_until_next']
        )
        
        # Combined loss - SIMPLIFIED
        total_loss = (
            self.w_products * loss_products +
            self.w_total_qty * loss_total_qty +
            self.w_num_skus * loss_num_skus +
            self.w_days * loss_days
        )
        
        loss_dict = {
            'products': loss_products.item(),
            'total_qty': loss_total_qty.item(),
            'num_skus': loss_num_skus.item(),
            'days': loss_days.item()
        }
        
        return total_loss, loss_dict


class TrainerV2:
    """Training orchestrator - SIMPLIFIED"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Optimizer: AdamW with LOWER LR
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LR * 0.3,  # REDUCED: 0.001 → 0.0003
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.criterion = MultiTaskLossV2()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_top5 = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_top5_acc': [],
            'val_top5_acc': [],
            'train_top10_acc': [],
            'val_top10_acc': [],
            'lr': []
        }
    
    def train_epoch(self):
        """Train 1 epoch"""
        self.model.train()
        total_loss = 0
        total_top5_correct = 0
        total_top10_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            predictions = self.model(
                batch['hist_products'],
                batch['hist_quantities'],
                batch['hist_total_qty'],
                batch['hist_num_skus'],
                batch['hist_days_between']
            )
            
            loss, loss_dict = self.criterion(predictions, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Top-5 and Top-10 accuracy
            _, top10_pred = predictions['products'].topk(10, dim=1)
            target_first = batch['target_products'][:, 0]
            
            for i in range(len(target_first)):
                pred_list = top10_pred[i].tolist()
                if target_first[i].item() in pred_list[:5]:
                    total_top5_correct += 1
                if target_first[i].item() in pred_list:
                    total_top10_correct += 1
            
            total_samples += len(target_first)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'top5': f"{total_top5_correct/total_samples:.3f}",
                'top10': f"{total_top10_correct/total_samples:.3f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        top5_acc = total_top5_correct / total_samples
        top10_acc = total_top10_correct / total_samples
        
        return avg_loss, top5_acc, top10_acc
    
    @torch.no_grad()
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        total_top5_correct = 0
        total_top10_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            predictions = self.model(
                batch['hist_products'],
                batch['hist_quantities'],
                batch['hist_total_qty'],
                batch['hist_num_skus'],
                batch['hist_days_between']
            )
            
            loss, _ = self.criterion(predictions, batch)
            total_loss += loss.item()
            
            _, top10_pred = predictions['products'].topk(10, dim=1)
            target_first = batch['target_products'][:, 0]
            
            for i in range(len(target_first)):
                pred_list = top10_pred[i].tolist()
                if target_first[i].item() in pred_list[:5]:
                    total_top5_correct += 1
                if target_first[i].item() in pred_list:
                    total_top10_correct += 1
            
            total_samples += len(target_first)
        
        avg_loss = total_loss / len(self.val_loader)
        top5_acc = total_top5_correct / total_samples
        top10_acc = total_top10_correct / total_samples
        
        return avg_loss, top5_acc, top10_acc
    
    def train(self):
        """Full training loop"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING START - TFT SIMPLIFIED (PRODUCTS FOCUS)")
        logger.info("="*80)
        
        for epoch in range(self.cfg.EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{self.cfg.EPOCHS}")
            logger.info("-" * 80)
            
            train_loss, train_top5, train_top10 = self.train_epoch()
            val_loss, val_top5, val_top10 = self.validate()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_top5_acc'].append(train_top5)
            self.history['val_top5_acc'].append(val_top5)
            self.history['train_top10_acc'].append(train_top10)
            self.history['val_top10_acc'].append(val_top10)
            self.history['lr'].append(current_lr)
            
            logger.info(f"Train Loss: {train_loss:.3f} | Top-5: {train_top5*100:.1f}% | Top-10: {train_top10*100:.1f}%")
            logger.info(f"Val Loss: {val_loss:.3f} | Top-5: {val_top5*100:.1f}% | Top-10: {val_top10*100:.1f}%")
            logger.info(f"LR: {current_lr:.6f}")
            
            # Save based on Top-5 accuracy (better metric than loss)
            if val_top5 > self.best_val_top5:
                improvement = val_top5 - self.best_val_top5
                self.best_val_top5 = val_top5
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"✅ Best model saved! Top-5 improved by {improvement*100:.2f}%")
            else:
                self.patience_counter += 1
                logger.info(f"⚠️ No improvement ({self.patience_counter}/{self.cfg.EARLY_STOP_PATIENCE})")
                
                if self.patience_counter >= self.cfg.EARLY_STOP_PATIENCE:
                    logger.info("🛑 Early stopping!")
                    break
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info(f"Best Val Top-5: {self.best_val_top5*100:.2f}%")
        logger.info(f"Best Val Loss: {self.best_val_loss:.3f}")
        logger.info("="*80)
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_top5': self.best_val_top5,
            'history': self.history
        }
        
        path = self.cfg.MODELS / ('best_model_v2.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
        logger.info(f"💾 Saved: {path.name}")


def main():
    """Main pipeline - SIMPLIFIED"""
    
    print("\n" + "="*80)
    print("CUSTOMER BEHAVIOR PREDICTION V2 - SIMPLIFIED")
    print("Focus: Products + Quantities (NO MONEY)")
    print("="*80 + "\n")
    
    SystemConfig.init_dirs()
    SystemConfig.summary()
    
    # Step 1: Data loading
    logger.info("[1/6] Data loading and preprocessing...")
    data_loader = DataLoaderV2(SystemConfig)
    train_orders, val_orders, test_orders = data_loader.prepare_pipeline()
    
    # Step 2: Fit normalization stats
    logger.info("\n[2/6] Fitting normalization statistics...")
    norm_stats = NormalizationStats()
    norm_stats.fit(train_orders)
    
    # Step 3: Sequence building
    logger.info("\n[3/6] Building sequences...")
    builder = SequenceBuilderV2(SystemConfig, norm_stats)
    
    train_seq = builder.build_all_sequences(train_orders)
    val_seq = builder.build_all_sequences(val_orders)
    test_seq = builder.build_all_sequences(test_orders)
    
    train_seq.to_pickle(SystemConfig.PROCESSED / 'train_sequences.pkl')
    val_seq.to_pickle(SystemConfig.PROCESSED / 'val_sequences.pkl')
    test_seq.to_pickle(SystemConfig.PROCESSED / 'test_sequences.pkl')
    
    # Step 4: Vocabulary
    logger.info("\n[4/6] Building vocabulary...")
    vocab = VocabularyBuilder()
    vocab.fit(train_seq)
    
    with open(SystemConfig.PROCESSED / 'vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(SystemConfig.PROCESSED / 'norm_stats.pkl', 'wb') as f:
        pickle.dump(norm_stats, f)
    
    # Step 5: Create dataloaders
    logger.info("\n[5/6] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders_v2(
        train_seq, val_seq, test_seq, vocab, SystemConfig
    )
    
    # Step 6: Training
    logger.info("\n[6/6] Training TFT model...")
    model = create_tft_model(SystemConfig, vocab.product_vocab_size)
    
    trainer = TrainerV2(model, train_loader, val_loader, SystemConfig)
    history = trainer.train()
    
    # Save final artifacts
    logger.info("\nSaving final artifacts...")
    
    with open(SystemConfig.MODELS / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocabulary': vocab,
        'norm_stats': norm_stats,
        'history': history,
        'config': {
            'hidden_dim': SystemConfig.HIDDEN_DIM,
            'num_heads': SystemConfig.NUM_HEADS,
            'vocab_size': vocab.product_vocab_size
        }
    }
    torch.save(final_checkpoint, SystemConfig.MODELS / 'final_model_v2.pth')
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"Best Top-5 accuracy: {trainer.best_val_top5*100:.2f}%")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.3f}")
    logger.info(f"Model saved to: {SystemConfig.MODELS}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise