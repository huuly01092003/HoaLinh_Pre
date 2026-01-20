"""
Time-Series Sales Forecasting - Production Pipeline
Proper time-based split for realistic forecasting
Run: python main_timeseries.py --env development
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import gc
import torch
from datetime import datetime, timedelta

# Force immediate output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("\n[TIMESERIES] Starting imports...", flush=True)

from config import get_config
from geocache import GeoCache
from features import FeatureEngineer
from model import create_timeseries_model
from data_manager import DataManager

print("[TIMESERIES] All imports successful!", flush=True)


def setup_logging(config):
    """Setup logging"""
    log_file = config.LOGS_DIR / f'timeseries_training_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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
    print(f"[SETUP_LOGGING] Logger initialized: {log_file}", flush=True)
    return logger


def time_based_split(df, config, logger):
    """
    CRITICAL: Split data by TIME, not randomly
    This prevents data leakage and creates realistic evaluation
    """
    logger.info("\n" + "="*80)
    logger.info("TIME-BASED DATA SPLIT (Anti-Leakage)")
    logger.info("="*80)
    
    # Convert date column
    df['NGÀY TẠO ĐƠN BÁN'] = pd.to_datetime(
        df['NGÀY TẠO ĐƠN BÁN'], 
        format='%d/%m/%Y', 
        errors='coerce'
    )
    
    # Remove invalid dates
    df = df.dropna(subset=['NGÀY TẠO ĐƠN BÁN'])
    
    # Get date range
    min_date = df['NGÀY TẠO ĐƠN BÁN'].min()
    max_date = df['NGÀY TẠO ĐƠN BÁN'].max()
    
    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
    logger.info(f"Total days: {(max_date - min_date).days}")
    
    # Define cutoffs (use 70% for train, 15% for val, 15% for test)
    total_days = (max_date - min_date).days
    train_days = int(total_days * 0.70)
    val_days = int(total_days * 0.15)
    
    cutoff_train = min_date + timedelta(days=train_days)
    cutoff_val = cutoff_train + timedelta(days=val_days)
    
    # Split
    train_df = df[df['NGÀY TẠO ĐƠN BÁN'] <= cutoff_train].copy()
    val_df = df[(df['NGÀY TẠO ĐƠN BÁN'] > cutoff_train) & 
                (df['NGÀY TẠO ĐƠN BÁN'] <= cutoff_val)].copy()
    test_df = df[df['NGÀY TẠO ĐƠN BÁN'] > cutoff_val].copy()
    
    logger.info(f"\n📅 TRAIN: {min_date.date()} to {cutoff_train.date()}")
    logger.info(f"   Rows: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"   Unique customers: {train_df['MÃ KHÁCH HÀNG'].nunique():,}")
    
    logger.info(f"\n📅 VAL: {(cutoff_train + timedelta(days=1)).date()} to {cutoff_val.date()}")
    logger.info(f"   Rows: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"   Unique customers: {val_df['MÃ KHÁCH HÀNG'].nunique():,}")
    
    logger.info(f"\n📅 TEST: {(cutoff_val + timedelta(days=1)).date()} to {max_date.date()}")
    logger.info(f"   Rows: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"   Unique customers: {test_df['MÃ KHÁCH HÀNG'].nunique():,}")
    
    # Validate no overlap
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df['NGÀY TẠO ĐƠN BÁN'].max() < val_df['NGÀY TẠO ĐƠN BÁN'].min()
    assert val_df['NGÀY TẠO ĐƠN BÁN'].max() < test_df['NGÀY TẠO ĐƠN BÁN'].min()
    
    logger.info("\n✅ Time-based split validated - NO DATA LEAKAGE")
    logger.info("="*80 + "\n")
    
    return train_df, val_df, test_df, cutoff_train, cutoff_val


def load_and_prepare_data(config, logger):
    """Load and prepare data with proper time-based split"""
    
    logger.info("\n[1/6] Loading raw data...")
    print("[DATA] Loading raw data...", flush=True)
    
    # Load in chunks for memory efficiency
    chunks = []
    for chunk in pd.read_csv(
        config.RAW_DATA_PATH,
        encoding='utf-8-sig',
        chunksize=200000,
        low_memory=False
    ):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[DATA] Loaded {len(df):,} rows", flush=True)
    
    # Map columns
    logger.info("\n[2/6] Mapping columns...")
    from column_mapper import map_columns
    df = map_columns(df)
    logger.info(f"✅ Using {len(df.columns)} columns")
    
    # CRITICAL: Clean numeric columns BEFORE splitting
    logger.info("\n[3/6] Cleaning numeric columns...")
    numeric_cols = [
        'SỐ LƯỢNG ĐƠN BÁN', 'ĐƠN GIÁ ĐƠN BÁN', 'TỔNG TIỀN ĐƠN BÁN',
        'CHIẾT KHẤU ĐƠN BÁN', 'THÀNH TIỀN ĐƠN BÁN'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove commas and convert
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.replace(' ', ''),
                errors='coerce'
            ).fillna(0)
            
            # Validate
            total = df[col].sum()
            logger.info(f"  {col}: Total = {total:,.0f}")
            
            if total == 0:
                logger.warning(f"  ⚠️ WARNING: {col} is all zeros!")
    
    # Time-based split
    logger.info("\n[4/6] Time-based split...")
    train_df, val_df, test_df, cutoff_train, cutoff_val = time_based_split(df, config, logger)
    
    return train_df, val_df, test_df


def create_timeseries_dataset(sequences_df, product_to_id, config, split_name, cutoff_date):
    """
    Create dataset with time awareness
    Each sample knows what date it's predicting for
    """
    import torch
    from torch.utils.data import Dataset
    
    class TimeSeriesDataset(Dataset):
        def __init__(self, sequences, product_to_id, max_seq_len, split_name, cutoff_date):
            self.sequences = sequences
            self.product_to_id = product_to_id
            self.max_seq_len = max_seq_len
            self.split_name = split_name
            self.cutoff_date = cutoff_date
            
            self.prepare_tensors()
        
        def _pad_sequence(self, seq, max_len, pad_value=0):
            seq = list(seq)
            if len(seq) > max_len:
                return seq[-max_len:]
            return [pad_value] * (max_len - len(seq)) + seq
        
        def prepare_tensors(self):
            # Same as before for product IDs, qty, revenue, discount
            self.product_ids = []
            for seq in self.sequences['product_seq']:
                ids = [self.product_to_id.get(p, 0) for p in seq]
                self.product_ids.append(self._pad_sequence(ids, self.max_seq_len, 0))
            self.product_ids = torch.tensor(self.product_ids, dtype=torch.long)
            
            self.qty = torch.tensor([
                self._pad_sequence(seq, self.max_seq_len, 0)
                for seq in self.sequences['qty_seq']
            ], dtype=torch.float)
            
            self.revenue = torch.tensor([
                self._pad_sequence(seq, self.max_seq_len, 0)
                for seq in self.sequences['revenue_seq']
            ], dtype=torch.float)
            
            self.discount = torch.tensor([
                self._pad_sequence(seq, self.max_seq_len, 0)
                for seq in self.sequences['discount_seq']
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
            
            # NEW: Time features for target prediction
            # For each sample, calculate what date we're predicting
            self.target_time_features = []
            
            for idx, row in self.sequences.iterrows():
                # Assuming next purchase is 7 days after last purchase (can adjust)
                days_until_prediction = 7  # Forecast horizon
                
                # Extract temporal features of target date
                target_dow = (row['day_of_week'] + days_until_prediction) % 7
                target_dom = min(31, row['day_of_month'] + days_until_prediction)  # Approximate
                target_month = row['month']  # Simplified
                
                self.target_time_features.append([
                    target_dow,
                    target_dom,
                    target_month,
                    days_until_prediction
                ])
            
            self.target_time_features = torch.tensor(
                self.target_time_features, dtype=torch.float
            )
            
            # Targets
            target_products = []
            for p in self.sequences['next_product']:
                pid = self.product_to_id.get(p, 0)
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
                'customer_features': self.customer_features[idx],
                'target_time_features': self.target_time_features[idx],
                'target_product': self.target_product[idx],
                'target_qty': self.target_qty[idx],
                'target_revenue': self.target_revenue[idx],
                'target_discount': self.target_discount[idx]
            }
    
    return TimeSeriesDataset(sequences_df, product_to_id, config.MAX_SEQ_LEN, 
                             split_name, cutoff_date)


def main(args):
    """Main time-series training pipeline"""
    
    print("\n[MAIN] Starting Time-Series Pipeline...", flush=True)
    
    # Load config
    config = get_config(args.env)
    config.create_directories()
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("\n" + "="*80)
    logger.info("TIME-SERIES SALES FORECASTING - PRODUCTION TRAINING")
    logger.info("="*80)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info("="*80 + "\n")
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data(config, logger)
    
    # Feature engineering
    logger.info("\n[5/6] Feature engineering...")
    cache_file = config.CACHE_DIR / 'geocache.pkl'
    geo_cache = GeoCache(cache_file=cache_file)
    
    logger.info("  Processing train set...")
    train_engineer = FeatureEngineer(train_df, config.CURRENT_DATE, geo_cache)
    train_engineer.process(fit_mode=True)
    train_sequences = train_engineer.prepare_sequences(config.MAX_SEQ_LEN)
    logger.info(f"  ✅ Train sequences: {len(train_sequences):,}")
    
    del train_df
    gc.collect()
    
    logger.info("  Processing val set...")
    val_engineer = FeatureEngineer(val_df, config.CURRENT_DATE, geo_cache)
    val_engineer.encoders = train_engineer.encoders
    val_engineer.process(fit_mode=False)
    val_sequences = val_engineer.prepare_sequences(config.MAX_SEQ_LEN)
    logger.info(f"  ✅ Val sequences: {len(val_sequences):,}")
    
    del val_df
    gc.collect()
    
    logger.info("  Processing test set...")
    test_engineer = FeatureEngineer(test_df, config.CURRENT_DATE, geo_cache)
    test_engineer.encoders = train_engineer.encoders
    test_engineer.process(fit_mode=False)
    test_sequences = test_engineer.prepare_sequences(config.MAX_SEQ_LEN)
    logger.info(f"  ✅ Test sequences: {len(test_sequences):,}")
    
    del test_df
    gc.collect()
    
    # Product mapping
    logger.info("\n[6/6] Creating product mapping...")
    unique_products = train_sequences['product_seq'].explode().unique()
    product_to_id = {p: i for i, p in enumerate(unique_products)}
    num_products = len(product_to_id)
    logger.info(f"✅ Total unique products: {num_products:,}")
    
    # Create datasets
    logger.info("\nCreating time-series datasets...")
    from torch.utils.data import DataLoader
    
    train_dataset = create_timeseries_dataset(
        train_sequences, product_to_id, config, 'train', None
    )
    val_dataset = create_timeseries_dataset(
        val_sequences, product_to_id, config, 'val', None
    )
    test_dataset = create_timeseries_dataset(
        test_sequences, product_to_id, config, 'test', None
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    logger.info(f"✅ Dataloaders ready:")
    logger.info(f"  Train batches: {len(train_loader):,}")
    logger.info(f"  Val batches: {len(val_loader):,}")
    logger.info(f"  Test batches: {len(test_loader):,}")
    
    # Create model
    logger.info("\nCreating time-series model...")
    model = create_timeseries_model(config, num_products)
    
    # Training (simplified - you can use your OptimizedTrainer with modifications)
    logger.info("\n✅ Data preparation complete!")
    logger.info("Next step: Implement training loop with time-aware loss")
    logger.info("\nKey improvements:")
    logger.info("  1. ✅ Time-based split (no data leakage)")
    logger.info("  2. ✅ Numeric columns properly cleaned")
    logger.info("  3. ✅ Time-aware model architecture")
    logger.info("  4. ⏳ Need to implement training loop")
    
    # Save checkpoint
    checkpoint = {
        'product_to_id': product_to_id,
        'num_products': num_products,
        'encoders': train_engineer.encoders,
        'config': config.to_dict()
    }
    
    checkpoint_path = config.MODEL_DIR / 'timeseries_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"\n💾 Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-Series Sales Forecasting')
    
    parser.add_argument('--env', type=str, default='development',
                       choices=['development', 'production', 'fast'])
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n[MAIN] ❌ FATAL ERROR: {e}", flush=True)
        logging.error(f"Training failed: {e}", exc_info=True)
        raise