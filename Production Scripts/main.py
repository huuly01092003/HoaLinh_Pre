"""
Full Dataset Training Script
Handles 2.4M rows efficiently with chunked processing and caching
Run: python main_full.py --env production
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import gc
import torch

from config import get_config
from geocache import GeoCache
from features import FeatureEngineer
from dataset import create_dataloaders
from model import create_model
from train import OptimizedTrainer, evaluate_model
from data_manager import DataManager


def setup_logging(config):
    """Setup comprehensive logging"""
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


def load_data_in_chunks(config, logger, chunk_size=200000):
    """
    Load large CSV in chunks to avoid memory issues
    Returns: complete DataFrame
    """
    logger.info(f"Loading data from {config.RAW_DATA_PATH}")
    logger.info(f"Chunk size: {chunk_size:,} rows")
    
    # Count total rows
    logger.info("Counting total rows...")
    total_rows = sum(1 for _ in open(config.RAW_DATA_PATH, encoding='utf-8-sig')) - 1
    logger.info(f"Total rows: {total_rows:,}")
    
    # Load in chunks
    chunks = []
    rows_loaded = 0
    
    logger.info("Reading data in chunks...")
    
    for i, chunk in enumerate(pd.read_csv(
        config.RAW_DATA_PATH,
        encoding='utf-8-sig',
        chunksize=chunk_size,
        low_memory=False
    )):
        chunks.append(chunk)
        rows_loaded += len(chunk)
        progress = (rows_loaded / total_rows) * 100
        logger.info(f"  Chunk {i+1}: {len(chunk):,} rows | Total: {rows_loaded:,} ({progress:.1f}%)")
        
        # Memory check
        if i % 5 == 0 and i > 0:
            gc.collect()
    
    # Combine chunks
    logger.info("Combining all chunks...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    return df


def optimize_dtypes(df, logger):
    """Optimize data types to reduce memory"""
    logger.info("Optimizing data types...")
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**3
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        col_type = df[col].dtype
        
        if col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        elif col_type == 'float64':
            df[col] = df[col].astype(np.float32)
    
    # Categorical columns with low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 1000:  # Low cardinality
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**3
    reduction = (start_mem - end_mem) / start_mem * 100
    
    logger.info(f"✅ Memory: {start_mem:.2f}GB → {end_mem:.2f}GB ({reduction:.1f}% reduction)")
    
    return df


def process_with_cache(config, logger, force_rebuild=False, version='latest'):
    """
    Process data with smart caching
    Returns: train_sequences, val_sequences, test_sequences, product_to_id, encoders
    """
    data_manager = DataManager(config.PROCESSED_DATA_PATH, use_compression=True)
    cache_file = config.CACHE_DIR / 'geocache.pkl'
    geo_cache = GeoCache(cache_file=cache_file)
    
    # Check cache
    if not force_rebuild:
        logger.info("Checking for cached processed data...")
        
        if data_manager.check_cache_validity(config.RAW_DATA_PATH, version, max_age_days=30):
            try:
                logger.info("✅ Loading from cache...")
                train_seq, val_seq, test_seq = data_manager.load_sequences(version)
                product_to_id, encoders = data_manager.load_mappings(version)
                
                logger.info("✅ Cache loaded successfully!")
                return train_seq, val_seq, test_seq, product_to_id, encoders
            
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}")
                logger.info("Rebuilding from scratch...")
    
    # Process from scratch
    logger.info("\n" + "="*80)
    logger.info("PROCESSING DATA FROM SCRATCH")
    logger.info("="*80 + "\n")
    
    # Step 1: Load data
    logger.info("[1/6] Loading data...")
    df = load_data_in_chunks(config, logger, chunk_size=200000)
    
    # Step 2: Optimize memory
    logger.info("\n[2/6] Optimizing memory...")
    df = optimize_dtypes(df, logger)
    
    # Step 3: Map columns
    logger.info("\n[3/6] Mapping columns...")
    from column_mapper import map_columns
    df = map_columns(df)
    logger.info(f"✅ Using {len(df.columns)} columns")
    
    # Step 4: Split data
    logger.info("\n[4/6] Splitting data...")
    logger.info("Train: 85%, Val: 4.5%, Test: 10.5%")
    
    train_df, temp_df = train_test_split(df, test_size=0.15, random_state=config.RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.7, random_state=config.RANDOM_STATE)
    
    logger.info(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    del df
    gc.collect()
    
    # Step 5: Feature engineering
    logger.info("\n[5/6] Feature engineering...")
    
    logger.info("  Processing train set...")
    train_engineer = FeatureEngineer(train_df, config.CURRENT_DATE, geo_cache)
    train_engineer.process(fit_mode=True)
    train_sequences = train_engineer.prepare_sequences(config.MAX_SEQ_LEN)
    logger.info(f"  ✅ Train sequences: {len(train_sequences):,}")
    
    del train_df
    gc.collect()
    
    logger.info("  Processing validation set...")
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
    
    # Step 6: Create product mapping
    logger.info("\n[6/6] Creating product mapping...")
    unique_products = train_sequences['product_seq'].explode().unique()
    product_to_id = {p: i+1 for i, p in enumerate(unique_products)}
    num_products = len(product_to_id)
    logger.info(f"✅ Total unique products: {num_products:,}")
    
    # Save to cache
    logger.info("\n💾 Saving to cache...")
    data_manager.save_sequences(train_sequences, val_sequences, test_sequences, version)
    data_manager.save_mappings(product_to_id, train_engineer.encoders, version)
    geo_cache.save_cache()
    
    # Show cache stats
    stats = data_manager.get_cache_stats()
    logger.info(f"\n📊 Cache statistics:")
    logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
    logger.info(f"  Files: {stats['file_count']}")
    logger.info(f"  Versions: {', '.join(stats['versions'])}")
    
    logger.info("\n✅ Data processing complete!\n")
    
    return train_sequences, val_sequences, test_sequences, product_to_id, train_engineer.encoders


def main(args):
    """Main training pipeline for full dataset"""
    
    # Load configuration
    config = get_config(args.env)
    config.create_directories()
    
    # Adjust batch size for large dataset
    if args.env == 'production':
        config.BATCH_SIZE = 128  # Larger batch for efficiency
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("\n" + "="*80)
    logger.info("SALES PREDICTION MODEL - FULL DATASET TRAINING")
    logger.info("="*80)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Workers: {config.NUM_WORKERS}")
    logger.info("="*80 + "\n")
    
    # Process data with caching
    train_sequences, val_sequences, test_sequences, product_to_id, encoders = process_with_cache(
        config, logger,
        force_rebuild=args.force_rebuild,
        version=args.version
    )
    
    num_products = len(product_to_id)
    
    logger.info("\n" + "="*80)
    logger.info("DATA SUMMARY")
    logger.info("="*80)
    logger.info(f"Train sequences: {len(train_sequences):,}")
    logger.info(f"Val sequences: {len(val_sequences):,}")
    logger.info(f"Test sequences: {len(test_sequences):,}")
    logger.info(f"Total products: {num_products:,}")
    logger.info("="*80 + "\n")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_sequences, val_sequences, test_sequences,
        product_to_id, config
    )
    
    # Clean up sequences to free memory
    del train_sequences, val_sequences, test_sequences
    gc.collect()
    
    logger.info(f"✅ Dataloaders ready:")
    logger.info(f"  Train batches: {len(train_loader):,}")
    logger.info(f"  Val batches: {len(val_loader):,}")
    logger.info(f"  Test batches: {len(test_loader):,}")
    
    # Create model
    logger.info("\n" + "="*80)
    logger.info("MODEL INITIALIZATION")
    logger.info("="*80)
    
    model = create_model(config, num_products)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    logger.info("="*80 + "\n")
    
    # Training with gradient accumulation
    accumulation_steps = args.accumulation_steps
    effective_batch = config.BATCH_SIZE * accumulation_steps
    
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Gradient accumulation steps: {accumulation_steps}")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Total batches per epoch: {len(train_loader):,}")
    logger.info(f"Optimizer steps per epoch: {len(train_loader) // accumulation_steps:,}")
    logger.info("="*80 + "\n")
    
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=config.MODEL_DIR,
        accumulation_steps=accumulation_steps
    )
    
    # Train
    history = trainer.train(epochs=config.EPOCHS)
    
    # Final evaluation on test set
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80 + "\n")
    
    best_model_path = config.MODEL_DIR / 'best_model.pth'
    trainer.load_checkpoint(best_model_path)
    
    test_metrics, test_preds, test_targets = evaluate_model(
        trainer.model, test_loader, config.DEVICE
    )
    
    # Save final artifacts
    logger.info("Saving final artifacts...")
    
    # Save metrics
    metrics_path = config.MODEL_DIR / 'test_metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(test_metrics, f)
    
    # Save complete checkpoint
    final_checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'product_to_id': product_to_id,
        'num_products': num_products,
        'config': config.to_dict(),
        'test_metrics': test_metrics,
        'history': history,
        'encoders': encoders
    }
    
    final_path = config.MODEL_DIR / 'final_model.pth'
    torch.save(final_checkpoint, final_path)
    
    logger.info(f"✅ Final model saved: {final_path}")
    logger.info(f"✅ Test metrics saved: {metrics_path}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Test accuracy: {test_metrics['product_accuracy']*100:.2f}%")
    logger.info(f"Test revenue MAE: {test_metrics['revenue_mae']:,.0f} VND")
    logger.info(f"Test revenue RMSE: {test_metrics['revenue_rmse']:,.0f} VND")
    logger.info(f"Total epochs trained: {len(history['train_loss'])}")
    logger.info("="*80)
    
    logger.info("\n✅ TRAINING COMPLETED SUCCESSFULLY! ✅\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on Full Dataset (2.4M rows)')
    
    parser.add_argument(
        '--env',
        type=str,
        default='production',
        choices=['development', 'production', 'experiment', 'fast'],
        help='Environment configuration (default: production)'
    )
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of processed data (ignore cache)'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='full_v1',
        help='Version name for processed data cache'
    )
    
    parser.add_argument(
        '--accumulation-steps',
        type=int,
        default=4,
        help='Gradient accumulation steps (default: 4)'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        raise