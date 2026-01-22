"""
PyTorch Dataset V2 - SIMPLIFIED
Focus: Products + Quantities (NO MONEY COLUMNS)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OrderSequenceDataset(Dataset):
    """
    Dataset cho order sequences - SIMPLIFIED
    BỎ: prices, order totals, line totals
    GIỮ: products, quantities, num_skus, days_between
    """
    
    def __init__(self, sequences_df, vocabulary, max_products_per_order=10):
        self.sequences = sequences_df
        self.vocab = vocabulary
        self.max_products = max_products_per_order
        
        self._prepare_tensors()
    
    def _pad_list(self, items_list, pad_value=0):
        """Pad list to max_products"""
        if len(items_list) > self.max_products:
            return items_list[:self.max_products]
        
        padding = [pad_value] * (self.max_products - len(items_list))
        return items_list + padding
    
    def _prepare_tensors(self):
        """Convert sequences to tensors - SIMPLIFIED"""
        logger.info("Preparing tensors (SIMPLIFIED)...")
        
        # ===== HISTORY =====
        
        # 1. History products: [batch, history_len, max_products_per_order]
        hist_products = []
        for products_lists in self.sequences['hist_products']:
            padded = []
            for products in products_lists:
                product_ids = self.vocab.transform_products_list(products)
                padded.append(self._pad_list(product_ids, pad_value=0))
            hist_products.append(padded)
        
        self.hist_products = torch.tensor(hist_products, dtype=torch.long)
        
        # 2. History quantities: [batch, history_len, max_products_per_order]
        hist_quantities = []
        for quantities_lists in self.sequences['hist_quantities']:
            padded = []
            for quantities in quantities_lists:
                if isinstance(quantities, list):
                    padded.append(self._pad_list(quantities, pad_value=0))
                else:
                    padded.append([quantities] + [0] * (self.max_products - 1))
            hist_quantities.append(padded)
        
        self.hist_quantities = torch.tensor(hist_quantities, dtype=torch.float32)
        
        # 3. History total quantity: [batch, history_len]
        self.hist_total_qty = torch.tensor(
            self.sequences['hist_total_qty'].tolist(),
            dtype=torch.float32
        )
        
        # 4. History num SKUs: [batch, history_len]
        self.hist_num_skus = torch.tensor(
            self.sequences['hist_num_skus'].tolist(),
            dtype=torch.float32
        )
        
        # 5. History days between: [batch, history_len]
        self.hist_days_between = torch.tensor(
            self.sequences['hist_days_between'].tolist(),
            dtype=torch.float32
        )
        
        # ===== TARGET =====
        
        # 1. Target products: [batch, max_products_per_order]
        target_products = []
        for products in self.sequences['target_products']:
            product_ids = self.vocab.transform_products_list(products)
            target_products.append(self._pad_list(product_ids, pad_value=0))
        
        self.target_products = torch.tensor(target_products, dtype=torch.long)
        
        # 2. Target quantities: [batch, max_products_per_order]
        target_quantities = []
        for quantities in self.sequences['target_quantities']:
            if isinstance(quantities, list):
                target_quantities.append(self._pad_list(quantities, pad_value=0))
            else:
                target_quantities.append([quantities] + [0] * (self.max_products - 1))
        
        self.target_quantities = torch.tensor(target_quantities, dtype=torch.float32)
        
        # 3. Target scalars
        self.target_total_qty = torch.tensor(
            self.sequences['target_total_qty'].values,
            dtype=torch.float32
        ).unsqueeze(1)
        
        self.target_num_skus = torch.tensor(
            self.sequences['target_num_skus'].values,
            dtype=torch.float32
        ).unsqueeze(1)
        
        self.days_until_next = torch.tensor(
            self.sequences['days_until_next'].values,
            dtype=torch.float32
        ).unsqueeze(1)
        
        logger.info(f"✅ Tensors prepared for {len(self.sequences):,} samples (SIMPLIFIED)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            # History (NO MONEY)
            'hist_products': self.hist_products[idx],
            'hist_quantities': self.hist_quantities[idx],
            'hist_total_qty': self.hist_total_qty[idx],
            'hist_num_skus': self.hist_num_skus[idx],
            'hist_days_between': self.hist_days_between[idx],
            
            # Targets (NO MONEY)
            'target_products': self.target_products[idx],
            'target_quantities': self.target_quantities[idx],
            'target_total_qty': self.target_total_qty[idx],
            'target_num_skus': self.target_num_skus[idx],
            'days_until_next': self.days_until_next[idx]
        }
    
    def get_customer_info(self, idx):
        """Get customer metadata"""
        row = self.sequences.iloc[idx]
        return {
            'customer_id': row['customer_id'],
            'employee_id': row['employee_id'],
            'route_id': row['route_id'],
            'last_date': row['last_hist_date'],
            'target_date': row['target_date']
        }


def create_dataloaders_v2(train_seq, val_seq, test_seq, vocab, config):
    """Factory function cho dataloaders"""
    logger.info("Creating dataloaders (SIMPLIFIED)...")
    
    train_dataset = OrderSequenceDataset(train_seq, vocab)
    val_dataset = OrderSequenceDataset(val_seq, vocab)
    test_dataset = OrderSequenceDataset(test_seq, vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    
    logger.info(f"✅ Dataloaders ready (SIMPLIFIED):")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset):,} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset):,} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset):,} samples)")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config_v2 import SystemConfig
    import pickle
    
    logging.basicConfig(level=logging.INFO)
    
    # Load sequences and vocab
    train_seq = pd.read_pickle(SystemConfig.PROCESSED / 'train_sequences.pkl')
    with open(SystemConfig.PROCESSED / 'vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Create dataset
    dataset = OrderSequenceDataset(train_seq, vocab)
    
    # Test getitem
    sample = dataset[0]
    print("\n✅ Dataset test (SIMPLIFIED):")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")
    
    # Test customer info
    info = dataset.get_customer_info(0)
    print(f"\nCustomer info: {info}")