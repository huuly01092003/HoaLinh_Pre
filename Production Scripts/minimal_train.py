"""
Minimal Training Script - No fancy features, just works
Run: python minimal_train.py
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import sys

print("="*80)
print("🚀 MINIMAL TRAINING SCRIPT")
print("="*80)

# Step 1: Load config
print("\n[1/8] Loading config...")
sys.stdout.flush()
from config import get_config
config = get_config('development')
config.create_directories()
print("✓ Config loaded")
sys.stdout.flush()

# Step 2: Check data
print("\n[2/8] Checking data file...")
if not config.RAW_DATA_PATH.exists():
    print(f"❌ Data file not found: {config.RAW_DATA_PATH}")
    sys.stdout.flush()
    print("\n💡 Expected location: data/raw/merged_2025.csv")
    sys.stdout.flush()
    sys.exit(1)

# Count rows
print("   Counting rows...", end=" ", flush=True)
sys.stdout.flush()
total_rows = sum(1 for _ in open(config.RAW_DATA_PATH, encoding='utf-8-sig')) - 1
print(f"{total_rows:,} rows")
sys.stdout.flush()
# Step 3: Load sample data
# Step 3: Load sample data
print("\n[3/8] Loading data sample...")
sys.stdout.flush()
SAMPLE_SIZE = 50000  # Start with 50k rows
print(f"   Loading {SAMPLE_SIZE:,} rows...")
sys.stdout.flush()

# Import column mapper
from column_mapper import map_columns, print_column_comparison

df_raw = pd.read_csv(
    config.RAW_DATA_PATH, 
    encoding='utf-8-sig',
    nrows=SAMPLE_SIZE,
    low_memory=False
)

# Show column comparison
print_column_comparison(df_raw)

# Map columns
df = map_columns(df_raw)

print(f"✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
sys.stdout.flush()
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
sys.stdout.flush()

# Step 4: Quick feature engineering
print("\n[4/8] Basic feature engineering...")
sys.stdout.flush()
from geocache import GeoCache
from features import FeatureEngineer

cache_file = config.CACHE_DIR / 'geocache.pkl'
geo_cache = GeoCache(cache_file=cache_file)

# Split first
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"   Train: {len(train_df):,}")
sys.stdout.flush()
print(f"   Val: {len(val_df):,}")
sys.stdout.flush()
print(f"   Test: {len(test_df):,}")
sys.stdout.flush()
# Process train
print("   Processing train set...")
sys.stdout.flush()
train_engineer = FeatureEngineer(train_df, config.CURRENT_DATE, geo_cache)
train_engineer.process(fit_mode=True)
print("   Creating sequences...")
sys.stdout.flush()
sys.stdout.flush()
train_sequences = train_engineer.prepare_sequences(config.MAX_SEQ_LEN)
print(f"   ✓ Train sequences: {len(train_sequences):,}")
sys.stdout.flush()
# Process val
print("   Processing val set...")
sys.stdout.flush()
val_engineer = FeatureEngineer(val_df, config.CURRENT_DATE, geo_cache)
val_engineer.encoders = train_engineer.encoders
val_engineer.process(fit_mode=False)
val_sequences = val_engineer.prepare_sequences(config.MAX_SEQ_LEN)
print(f"   ✓ Val sequences: {len(val_sequences):,}")
sys.stdout.flush()
# Process test
print("   Processing test set...")
sys.stdout.flush()
test_engineer = FeatureEngineer(test_df, config.CURRENT_DATE, geo_cache)
test_engineer.encoders = train_engineer.encoders
test_engineer.process(fit_mode=False)
test_sequences = test_engineer.prepare_sequences(config.MAX_SEQ_LEN)
print(f"   ✓ Test sequences: {len(test_sequences):,}")
sys.stdout.flush()
# Step 5: Product mapping
print("\n[5/8] Creating product mapping...")
sys.stdout.flush()
unique_products = train_sequences['product_seq'].explode().unique()
product_to_id = {p: i+1 for i, p in enumerate(unique_products)}
num_products = len(product_to_id)
print(f"✓ {num_products:,} unique products")
sys.stdout.flush()
# Step 6: Create dataloaders
print("\n[6/8] Creating dataloaders...")
sys.stdout.flush()
from dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    train_sequences, val_sequences, test_sequences,
    product_to_id, config
)
print(f"✓ Train batches: {len(train_loader)}")
sys.stdout.flush()
print(f"✓ Val batches: {len(val_loader)}")
sys.stdout.flush()
print(f"✓ Test batches: {len(test_loader)}")
sys.stdout.flush()
# Step 7: Create model
print("\n[7/8] Creating model...")
sys.stdout.flush()
from model import create_model

model = create_model(config, num_products)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model ready: {total_params:,} parameters")
sys.stdout.flush()
# Step 8: Train
print("\n[8/8] Training...")
sys.stdout.flush()
from train import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    save_dir=config.MODEL_DIR
)

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Epochs: {config.EPOCHS}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Device: {config.DEVICE}")
print("="*80 + "\n")

sys.stdout.flush()

# Train!
history = trainer.train(epochs=config.EPOCHS)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
print(f"Model saved: {config.MODEL_DIR / 'best_model.pth'}")
print("="*80)