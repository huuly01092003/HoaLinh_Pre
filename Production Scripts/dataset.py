"""
PyTorch Dataset for Sales Sequences
Windows-optimized version (NUM_WORKERS=0)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict


class SalesSequenceDataset(Dataset):
    """Dataset for customer purchase sequences"""
    
    def __init__(
        self,
        sequences_df: pd.DataFrame,
        product_to_id: Dict[str, int],
        max_seq_len: int
    ):
        self.sequences = sequences_df
        self.product_to_id = product_to_id
        self.max_seq_len = max_seq_len
        
        self.prepare_tensors()
    
    def _pad_sequence(self, seq, max_len, pad_value=0):
        """Pad or truncate sequence to max_len"""
        seq = list(seq)
        if len(seq) > max_len:
            return seq[-max_len:]
        return [pad_value] * (max_len - len(seq)) + seq
    
    def prepare_tensors(self):
        """Convert sequences to padded tensors"""
        
        # Product IDs
        self.product_ids = []
        for seq in self.sequences['product_seq']:
            ids = [self.product_to_id.get(p, 0) for p in seq]
            self.product_ids.append(self._pad_sequence(ids, self.max_seq_len, 0))
        self.product_ids = torch.tensor(self.product_ids, dtype=torch.long)
        
        # Quantity sequence
        self.qty = torch.tensor([
            self._pad_sequence(seq, self.max_seq_len, 0)
            for seq in self.sequences['qty_seq']
        ], dtype=torch.float)
        
        # Revenue sequence
        self.revenue = torch.tensor([
            self._pad_sequence(seq, self.max_seq_len, 0)
            for seq in self.sequences['revenue_seq']
        ], dtype=torch.float)
        
        # Discount sequence
        self.discount = torch.tensor([
            self._pad_sequence(seq, self.max_seq_len, 0)
            for seq in self.sequences['discount_seq']
        ], dtype=torch.float)
        
        # Temporal changes
        self.week_change = torch.tensor([
            self._pad_sequence(seq, self.max_seq_len, 0)
            for seq in self.sequences['week_change_seq']
        ], dtype=torch.float)
        
        self.month_change = torch.tensor([
            self._pad_sequence(seq, self.max_seq_len, 0)
            for seq in self.sequences['month_change_seq']
        ], dtype=torch.float)
        
        self.quarter_change = torch.tensor([
            self._pad_sequence(seq, self.max_seq_len, 0)
            for seq in self.sequences['quarter_change_seq']
        ], dtype=torch.float)
        
        self.year_change = torch.tensor([
            self._pad_sequence(seq, self.max_seq_len, 0)
            for seq in self.sequences['year_change_seq']
        ], dtype=torch.float)
        
        # Customer features
        customer_feature_cols = [
            'recency', 'frequency', 'monetary', 'customer_lifetime',
            'num_unique_products', 'avg_discount', 'distance_to_employee',
            'customer_segment', 'is_walkin', 'is_weekend',
            'hour', 'day_of_week', 'month', 'quarter'
        ]
        
        customer_features = self.sequences[customer_feature_cols].fillna(0)
        self.customer_features = torch.tensor(
            customer_features.values, dtype=torch.float
        )
        
        # Targets - Map products to IDs
        target_products = []
        max_valid_id = len(self.product_to_id) - 1  # 0-indexed

        for p in self.sequences['next_product']:
            pid = self.product_to_id.get(p, 0)  # Default to 0 if unknown
            
            # Safety check: ensure ID is in valid rangze [0, num_products-1]
            if pid < 0 or pid > max_valid_id:
                print(f"WARNING: Invalid product ID {pid} for product {p}, using 0")
                pid = 0
            
            target_products.append(pid)

        self.target_product = torch.tensor(target_products, dtype=torch.long)
        
        self.target_qty = torch.tensor(
            self.sequences['next_quantity'].fillna(0).values,
            dtype=torch.float
        ).unsqueeze(1)
        
        self.target_revenue = torch.tensor(
            self.sequences['next_revenue'].fillna(0).values,
            dtype=torch.float
        ).unsqueeze(1)
        
        self.target_discount = torch.tensor(
            self.sequences['next_discount'].fillna(0).values,
            dtype=torch.float
        ).unsqueeze(1)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'product_ids': self.product_ids[idx],
            'qty': self.qty[idx],
            'revenue': self.revenue[idx],
            'discount': self.discount[idx],
            'week_change': self.week_change[idx],
            'month_change': self.month_change[idx],
            'quarter_change': self.quarter_change[idx],
            'year_change': self.year_change[idx],
            'customer_features': self.customer_features[idx],
            'target_product': self.target_product[idx],
            'target_qty': self.target_qty[idx],
            'target_revenue': self.target_revenue[idx],
            'target_discount': self.target_discount[idx]
        }
    
    def get_customer_ids(self):
        """Get customer IDs for analysis"""
        return self.sequences['customer_id'].tolist()


def create_dataloaders(
    train_sequences: pd.DataFrame,
    val_sequences: pd.DataFrame,
    test_sequences: pd.DataFrame,
    product_to_id: Dict[str, int],
    config
):
    """
    Factory function to create train/val/test dataloaders
    Windows-optimized: NUM_WORKERS=0 for better compatibility
    """
    
    train_dataset = SalesSequenceDataset(
        train_sequences, product_to_id, config.MAX_SEQ_LEN
    )
    val_dataset = SalesSequenceDataset(
        val_sequences, product_to_id, config.MAX_SEQ_LEN
    )
    test_dataset = SalesSequenceDataset(
        test_sequences, product_to_id, config.MAX_SEQ_LEN
    )
    
    # Windows optimization: num_workers=0, pin_memory=False
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # ← Windows works better with 0
        pin_memory=False,  # ← False for CPU
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    print(f"✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader