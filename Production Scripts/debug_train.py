"""
Debug Training - Test each step separately
Run: python debug_train.py
"""

import sys
import traceback
from config import get_config
def test_step(step_num, description, func):
    """Test a single step and catch errors"""
    print(f"\n{'='*80}")
    print(f"[STEP {step_num}] {description}")
    print('='*80)
    sys.stdout.flush()
    
    try:
        result = func()
        print(f"✅ SUCCESS: {description}")
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"❌ FAILED: {description}")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


# STEP 1: Import config
def step1_import_config():
    print("Importing config...")
    
    config = get_config('development')
    config.create_directories()
    print(f"  Config loaded: {config.BATCH_SIZE} batch size")
    return config

config = test_step(1, "Import Config", step1_import_config)


# STEP 2: Check data file
def step2_check_data():
    print(f"Checking data file: {config.RAW_DATA_PATH}")
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {config.RAW_DATA_PATH}")
    
    # Count rows
    print("  Counting rows...", end=" ", flush=True)
    row_count = sum(1 for _ in open(config.RAW_DATA_PATH, encoding='utf-8-sig')) - 1
    print(f"{row_count:,} rows")
    return row_count

row_count = test_step(2, "Check Data File", step2_check_data)


# STEP 3: Load sample data
def step3_load_sample():
    import pandas as pd
    print("Loading 10,000 rows...")
    df = pd.read_csv(
        config.RAW_DATA_PATH,
        encoding='utf-8-sig',
        nrows=10000,
        low_memory=False
    )
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  First columns: {list(df.columns[:5])}")
    return df

df = test_step(3, "Load Sample Data", step3_load_sample)


# STEP 4: Map columns
def step4_map_columns():
    from column_mapper import map_columns, print_column_comparison
    print("Mapping columns...")
    print_column_comparison(df)
    df_mapped = map_columns(df)
    print(f"  Mapped columns: {len(df_mapped.columns)}")
    return df_mapped

df_mapped = test_step(4, "Map Columns", step4_map_columns)


# STEP 5: Split data
def step5_split_data():
    from sklearn.model_selection import train_test_split
    print("Splitting data...")
    train_df, temp_df = train_test_split(df_mapped, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    return train_df, val_df, test_df

train_df, val_df, test_df = test_step(5, "Split Data", step5_split_data)


# STEP 6: Initialize geocache
def step6_init_geocache():
    from geocache import GeoCache
    print("Initializing geocache...")
    cache_file = config.CACHE_DIR / 'geocache.pkl'
    geo_cache = GeoCache(cache_file=cache_file)
    print(f"  Geocache ready")
    return geo_cache

geo_cache = test_step(6, "Initialize Geocache", step6_init_geocache)


# STEP 7: Feature engineering on TRAIN
def step7_process_train():
    from features import FeatureEngineer
    print("Processing TRAIN features...")
    print(f"  Input rows: {len(train_df):,}")
    
    engineer = FeatureEngineer(train_df, config.CURRENT_DATE, geo_cache)
    
    print("  Running process()...")
    engineer.process(fit_mode=True)
    
    print("  Creating sequences...")
    sequences = engineer.prepare_sequences(config.MAX_SEQ_LEN)
    
    print(f"  Output sequences: {len(sequences):,}")
    return engineer, sequences

train_engineer, train_sequences = test_step(7, "Process Train Features", step7_process_train)


# STEP 8: Feature engineering on VAL
def step8_process_val():
    from features import FeatureEngineer
    print("Processing VAL features...")
    print(f"  Input rows: {len(val_df):,}")
    
    engineer = FeatureEngineer(val_df, config.CURRENT_DATE, geo_cache)
    engineer.encoders = train_engineer.encoders
    
    print("  Running process()...")
    engineer.process(fit_mode=False)
    
    print("  Creating sequences...")
    sequences = engineer.prepare_sequences(config.MAX_SEQ_LEN)
    
    print(f"  Output sequences: {len(sequences):,}")
    return sequences

val_sequences = test_step(8, "Process Val Features", step8_process_val)


# STEP 9: Feature engineering on TEST
def step9_process_test():
    from features import FeatureEngineer
    print("Processing TEST features...")
    print(f"  Input rows: {len(test_df):,}")
    
    engineer = FeatureEngineer(test_df, config.CURRENT_DATE, geo_cache)
    engineer.encoders = train_engineer.encoders
    
    print("  Running process()...")
    engineer.process(fit_mode=False)
    
    print("  Creating sequences...")
    sequences = engineer.prepare_sequences(config.MAX_SEQ_LEN)
    
    print(f"  Output sequences: {len(sequences):,}")
    return sequences

test_sequences = test_step(9, "Process Test Features", step9_process_test)


# STEP 10: Create product mapping - FIXED VERSION
def step10_product_mapping():
    print("Creating product mapping (FIXED)...")
    
    # Get ALL products from ORIGINAL train_df (before feature engineering)
    unique_products = train_df['MÃ SẢN PHẨM ĐƠN BÁN'].unique()
    
    # Create mapping with PAD token
    product_to_id = {'<PAD>': 0}
    product_to_id.update({p: i+1 for i, p in enumerate(unique_products)})
    
    num_products = len(unique_products)
    
    print(f"  Unique products: {num_products:,}")
    print(f"  Total vocab (with PAD): {len(product_to_id):,}")
    
    # Verify coverage
    train_prods = set(train_sequences['product_seq'].explode().unique())
    val_prods = set(val_sequences['product_seq'].explode().unique())
    test_prods = set(test_sequences['product_seq'].explode().unique())
    
    all_prods = train_prods | val_prods | test_prods
    known_prods = set(product_to_id.keys())
    
    unknown = all_prods - known_prods
    if unknown:
        print(f"  ⚠️  {len(unknown)} unknown products will map to PAD (ID=0)")
    else:
        print(f"  ✅ All products covered!")
    
    return product_to_id, num_products

product_to_id, num_products = test_step(10, "Create Product Mapping", step10_product_mapping)


# STEP 11: Create dataloaders
def step11_create_dataloaders():
    from dataset import create_dataloaders
    print("Creating dataloaders...")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_sequences, val_sequences, test_sequences,
        product_to_id, config
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = test_step(11, "Create Dataloaders", step11_create_dataloaders)


# STEP 12: Create model
def step12_create_model():
    from model import create_model
    print("Creating model...")
    model = create_model(config, num_products)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    return model

model = test_step(12, "Create Model", step12_create_model)


# STEP 13: Create trainer
def step13_create_trainer():
    from train import OptimizedTrainer
    print("Creating trainer...")
    
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=config.MODEL_DIR,
        accumulation_steps=2
    )
    
    print("  Trainer ready")
    return trainer

trainer = test_step(13, "Create Trainer", step13_create_trainer)


# STEP 14: Train ONE epoch
def step14_train_one_epoch():
    print("Training ONE epoch...")
    print("  This will take a few minutes...")
    
    train_loss, train_acc = trainer.train_epoch()
    
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Train accuracy: {train_acc*100:.2f}%")
    
    return train_loss, train_acc

train_loss, train_acc = test_step(14, "Train One Epoch", step14_train_one_epoch)


# STEP 15: Validate
def step15_validate():
    print("Validating...")
    
    val_loss, val_acc, val_mae, val_rmse = trainer.validate()
    
    print(f"  Val loss: {val_loss:.4f}")
    print(f"  Val accuracy: {val_acc*100:.2f}%")
    print(f"  Revenue MAE: {val_mae:,.0f}")
    print(f"  Revenue RMSE: {val_rmse:,.0f}")
    
    return val_loss, val_acc

val_loss, val_acc = test_step(15, "Validate", step15_validate)


# Final summary
print("\n" + "="*80)
print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nData: {row_count:,} rows")
print(f"Train sequences: {len(train_sequences):,}")
print(f"Products: {num_products:,}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"\nFirst epoch results:")
print(f"  Train Loss: {train_loss:.4f} | Accuracy: {train_acc*100:.2f}%")
print(f"  Val Loss: {val_loss:.4f} | Accuracy: {val_acc*100:.2f}%")
print("\n✅ Your setup is working! You can now run full training.")
print("="*80)