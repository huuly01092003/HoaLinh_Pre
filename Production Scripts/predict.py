"""
Predict next purchase for a single customer
Usage: python predict_customer.py --customer "KH001"
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from config import get_config
from geocache import GeoCache
from features import FeatureEngineer


def load_model_and_mappings(config):
    """Load trained model and mappings"""
    print("\n" + "="*80)
    print("LOADING MODEL AND MAPPINGS")
    print("="*80)
    
    # Load final model
    model_path = config.MODEL_DIR / 'final_model.pth'
    
    if not model_path.exists():
        # Try best model
        model_path = config.MODEL_DIR / 'best_model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {config.MODEL_DIR}")
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    # Extract components
    product_to_id = checkpoint.get('product_to_id')
    encoders = checkpoint.get('encoders')
    num_products = checkpoint.get('num_products', len(product_to_id))
    
    # Create model
    from model import create_model
    model = create_model(config, num_products)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create reverse mapping
    id_to_product = {v: k for k, v in product_to_id.items()}
    
    print(f"✅ Model loaded successfully!")
    print(f"   Total products: {num_products}")
    print(f"   Device: {config.DEVICE}")
    
    return model, product_to_id, id_to_product, encoders


def get_customer_data(customer_id, raw_data_path):
    """Get customer's purchase history"""
    print(f"\n{'='*80}")
    print(f"LOADING CUSTOMER DATA: {customer_id}")
    print("="*80)
    
    # Load raw data
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path, encoding='utf-8-sig', low_memory=False)
    
    # Filter customer
    from column_mapper import map_columns
    df = map_columns(df)
    
    customer_df = df[df['MÃ KHÁCH HÀNG'] == customer_id].copy()
    
    if len(customer_df) == 0:
        raise ValueError(f"Customer {customer_id} not found in dataset!")
    
    # Convert numeric columns
    numeric_cols = ['THÀNH TIỀN ĐƠN BÁN', 'SỐ LƯỢNG ĐƠN BÁN', 'ĐƠN GIÁ ĐƠN BÁN', 
                    'CHIẾT KHẤU ĐƠN BÁN', 'TỔNG TIỀN ĐƠN BÁN']
    for col in numeric_cols:
        if col in customer_df.columns:
            customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce').fillna(0)
    
    print(f"✅ Found {len(customer_df)} orders for customer {customer_id}")
    print(f"   Date range: {customer_df['NGÀY TẠO ĐƠN BÁN'].min()} to {customer_df['NGÀY TẠO ĐƠN BÁN'].max()}")
    print(f"   Total revenue: {customer_df['THÀNH TIỀN ĐƠN BÁN'].sum():,.0f} VND")
    print(f"   Unique products: {customer_df['MÃ SẢN PHẨM ĐƠN BÁN'].nunique()}")
    
    return customer_df


def prepare_customer_features(customer_df, config, geo_cache, encoders):
    """Process customer data into features"""
    print(f"\n{'='*80}")
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Feature engineering
    engineer = FeatureEngineer(customer_df, config.CURRENT_DATE, geo_cache)
    engineer.encoders = encoders
    engineer.process(fit_mode=False)
    
    # Create sequence
    sequences = engineer.prepare_sequences(config.MAX_SEQ_LEN)
    
    if len(sequences) == 0:
        raise ValueError("Failed to create sequence for customer!")
    
    print(f"✅ Features created successfully!")
    
    return sequences.iloc[0]  # Get first (only) sequence


def create_input_tensors(sequence, product_to_id, config):
    """Convert sequence to model inputs"""
    
    def pad_sequence(seq, max_len, pad_value=0):
        seq = list(seq)
        if len(seq) > max_len:
            return seq[-max_len:]
        return [pad_value] * (max_len - len(seq)) + seq
    
    # Product IDs
    product_ids = [product_to_id.get(p, 0) for p in sequence['product_seq']]
    product_ids = pad_sequence(product_ids, config.MAX_SEQ_LEN, 0)
    
    # Quantities
    qty = pad_sequence(sequence['qty_seq'], config.MAX_SEQ_LEN, 0)
    
    # Revenue
    revenue = pad_sequence(sequence['revenue_seq'], config.MAX_SEQ_LEN, 0)
    
    # Discount
    discount = pad_sequence(sequence['discount_seq'], config.MAX_SEQ_LEN, 0)
    
    # Temporal changes
    week_change = pad_sequence(sequence['week_change_seq'], config.MAX_SEQ_LEN, 0)
    month_change = pad_sequence(sequence['month_change_seq'], config.MAX_SEQ_LEN, 0)
    quarter_change = pad_sequence(sequence['quarter_change_seq'], config.MAX_SEQ_LEN, 0)
    year_change = pad_sequence(sequence['year_change_seq'], config.MAX_SEQ_LEN, 0)
    
    # Customer features
    customer_features = [
        sequence['recency'],
        sequence['frequency'],
        sequence['monetary'],
        sequence['customer_lifetime'],
        sequence['num_unique_products'],
        sequence['avg_discount'],
        sequence['distance_to_employee'],
        sequence['customer_segment'],
        sequence['is_walkin'],
        sequence['is_weekend'],
        sequence['hour'],
        sequence['day_of_week'],
        sequence['month'],
        sequence['quarter']
    ]
    
    # Convert to tensors
    inputs = {
        'product_ids': torch.tensor([product_ids], dtype=torch.long),
        'qty': torch.tensor([qty], dtype=torch.float).unsqueeze(-1),
        'revenue': torch.tensor([revenue], dtype=torch.float).unsqueeze(-1),
        'discount': torch.tensor([discount], dtype=torch.float).unsqueeze(-1),
        'week_change': torch.tensor([week_change], dtype=torch.float).unsqueeze(-1),
        'month_change': torch.tensor([month_change], dtype=torch.float).unsqueeze(-1),
        'quarter_change': torch.tensor([quarter_change], dtype=torch.float).unsqueeze(-1),
        'year_change': torch.tensor([year_change], dtype=torch.float).unsqueeze(-1),
        'customer_features': torch.tensor([customer_features], dtype=torch.float)
    }
    
    return inputs


def predict_next_purchase(model, inputs, id_to_product, config, top_k=5):
    """Make prediction"""
    print(f"\n{'='*80}")
    print("MAKING PREDICTION")
    print("="*80)
    
    # Move to device
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(
            inputs['product_ids'],
            inputs['qty'].squeeze(-1),
            inputs['revenue'].squeeze(-1),
            inputs['discount'].squeeze(-1),
            inputs['week_change'].squeeze(-1),
            inputs['month_change'].squeeze(-1),
            inputs['quarter_change'].squeeze(-1),
            inputs['year_change'].squeeze(-1),
            inputs['customer_features']
        )
    
    # Get predictions
    product_probs = torch.softmax(outputs['product'], dim=1)[0]
    predicted_qty = outputs['quantity'][0].item()
    predicted_revenue = outputs['revenue'][0].item()
    predicted_discount = outputs['discount'][0].item()
    
    # Top K products
    top_probs, top_indices = torch.topk(product_probs, min(top_k, len(product_probs)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        product_code = id_to_product.get(idx.item(), 'UNKNOWN')
        predictions.append({
            'product': product_code,
            'probability': prob.item() * 100,
            'predicted_qty': max(1, int(predicted_qty)),
            'predicted_revenue': max(0, predicted_revenue),
            'predicted_discount': max(0, min(1, predicted_discount))
        })
    
    return predictions


def display_results(customer_id, sequence, predictions):
    """Display prediction results"""
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    print(f"\nCustomer ID: {customer_id}")
    print(f"Purchase History: {len(sequence['product_seq'])} orders")
    print(f"RFM Metrics:")
    print(f"  - Recency: {sequence['recency']} days")
    print(f"  - Frequency: {sequence['frequency']:.0f} orders")
    print(f"  - Monetary: {sequence['monetary']:,.0f} VND")
    print(f"  - Customer Lifetime: {sequence['customer_lifetime']} days")
    print(f"  - Unique Products: {sequence['num_unique_products']}")
    
    print(f"\nLast 5 purchases:")
    for i, (prod, qty, rev) in enumerate(zip(
        sequence['product_seq'][-5:],
        sequence['qty_seq'][-5:],
        sequence['revenue_seq'][-5:]
    ), 1):
        print(f"  {i}. {prod} - Qty: {qty:.0f}, Revenue: {rev:,.0f} VND")
    
    print(f"\n{'='*80}")
    print("TOP PREDICTIONS FOR NEXT PURCHASE")
    print("="*80)
    
    for i, pred in enumerate(predictions, 1):
        print(f"\n{i}. Product: {pred['product']}")
        print(f"   Confidence: {pred['probability']:.2f}%")
        print(f"   Predicted Quantity: {pred['predicted_qty']}")
        print(f"   Predicted Revenue: {pred['predicted_revenue']:,.0f} VND")
        print(f"   Predicted Discount: {pred['predicted_discount']*100:.1f}%")
    
    print("\n" + "="*80)


def main():
    """Main prediction pipeline"""
    parser = argparse.ArgumentParser(description='Predict next purchase for a customer')
    parser.add_argument('--customer', type=str, required=True, help='Customer ID')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    parser.add_argument('--env', type=str, default='development', help='Environment config')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SALES PREDICTION - SINGLE CUSTOMER")
    print("="*80)
    print(f"Customer ID: {args.customer}")
    print(f"Top K predictions: {args.top_k}")
    print("="*80)
    
    # Load config
    config = get_config(args.env)
    
    # Load model and mappings
    model, product_to_id, id_to_product, encoders = load_model_and_mappings(config)
    
    # Load customer data
    customer_df = get_customer_data(args.customer, config.RAW_DATA_PATH)
    
    # Prepare features
    cache_file = config.CACHE_DIR / 'geocache.pkl'
    geo_cache = GeoCache(cache_file=cache_file)
    
    sequence = prepare_customer_features(customer_df, config, geo_cache, encoders)
    
    # Create input tensors
    inputs = create_input_tensors(sequence, product_to_id, config)
    
    # Make prediction
    predictions = predict_next_purchase(model, inputs, id_to_product, config, args.top_k)
    
    # Display results
    display_results(args.customer, sequence, predictions)
    
    # Save results
    results_dir = config.LOGS_DIR / 'predictions'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f'prediction_{args.customer}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Customer: {args.customer}\n")
        f.write(f"Prediction Date: {datetime.now()}\n\n")
        f.write("Top Predictions:\n")
        for i, pred in enumerate(predictions, 1):
            f.write(f"{i}. {pred['product']} - {pred['probability']:.2f}% confidence\n")
            f.write(f"   Qty: {pred['predicted_qty']}, Revenue: {pred['predicted_revenue']:,.0f} VND\n")
    
    print(f"\n✅ Results saved to: {results_file}")


if __name__ == "__main__":
    main()