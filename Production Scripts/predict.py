"""
Sales Prediction Script
Dự đoán sản phẩm tiếp theo cho khách hàng
Run: python predict.py --customer-id "C68N1000574"
"""

import argparse
import torch
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import sys

from config import get_config
from model import SalesPredictionModel
from geocache import GeoCache
from features import FeatureEngineer


def load_model(model_path: Path, config):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    # Get model info
    num_products = checkpoint['num_products']
    product_to_id = checkpoint['product_to_id']
    id_to_product = {v: k for k, v in product_to_id.items()}
    
    # Create model
    from model import create_model
    model = create_model(config, num_products)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Products: {num_products:,}")
    
    return model, product_to_id, id_to_product


def get_customer_history(customer_id: str, data_path: Path, max_rows: int = 100000):
    """Lấy lịch sử mua hàng của khách"""
    print(f"\nLoading customer history for: {customer_id}")
    
    # Load data (có thể load toàn bộ hoặc sample)
    df = pd.read_csv(data_path, encoding='utf-8-sig', nrows=max_rows, low_memory=False)
    
    # Filter customer
    customer_df = df[df['MÃ KHÁCH HÀNG'] == customer_id].copy()
    
    if len(customer_df) == 0:
        print(f"⚠️  Customer {customer_id} not found!")
        return None
    
    print(f"✓ Found {len(customer_df)} orders")
    
    return customer_df


def prepare_customer_features(customer_df: pd.DataFrame, config, geo_cache):
    """Chuẩn bị features cho dự đoán"""
    print("Preparing features...")
    
    from column_mapper import map_columns
    
    # Map columns
    customer_df = map_columns(customer_df)
    
    # Feature engineering
    engineer = FeatureEngineer(customer_df, config.CURRENT_DATE, geo_cache)
    engineer.process(fit_mode=False)
    
    # Create sequence
    sequence = engineer.prepare_sequences(config.MAX_SEQ_LEN)
    
    if len(sequence) == 0:
        print("⚠️  Cannot create sequence!")
        return None
    
    return sequence.iloc[-1]  # Get latest sequence


def predict_next_purchase(model, sequence, product_to_id, id_to_product, config, top_k=5):
    """Dự đoán sản phẩm tiếp theo"""
    print("\n" + "="*80)
    print("MAKING PREDICTION")
    print("="*80)
    
    # Prepare input tensors
    from dataset import SalesSequenceDataset
    
    # Create mini dataset with 1 sample
    seq_df = pd.DataFrame([sequence])
    
    dataset = SalesSequenceDataset(seq_df, product_to_id, config.MAX_SEQ_LEN)
    
    # Get the single sample
    sample = dataset[0]
    
    # Add batch dimension
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    
    # Predict
    with torch.no_grad():
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
    
    # Get predictions
    product_probs = torch.softmax(outputs['product'], dim=1)[0]
    predicted_qty = outputs['quantity'][0].item()
    predicted_revenue = outputs['revenue'][0].item()
    predicted_discount = outputs['discount'][0].item()
    
    # Top K products
    top_probs, top_indices = torch.topk(product_probs, k=min(top_k, len(product_probs)))
    
    # Display results
    print("\n🎯 TOP PREDICTED PRODUCTS:")
    print("-" * 80)
    
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        product_name = id_to_product.get(idx.item(), 'UNKNOWN')
        print(f"{i}. {product_name}")
        print(f"   Probability: {prob.item()*100:.2f}%")
        print(f"   Expected Quantity: {predicted_qty:.1f}")
        print(f"   Expected Revenue: {predicted_revenue:,.0f} VND")
        print(f"   Expected Discount: {predicted_discount*100:.1f}%")
        print()
    
    return {
        'top_products': [id_to_product.get(idx.item(), 'UNKNOWN') for idx in top_indices],
        'probabilities': top_probs.tolist(),
        'quantity': predicted_qty,
        'revenue': predicted_revenue,
        'discount': predicted_discount
    }


def show_customer_summary(customer_df: pd.DataFrame):
    """Hiển thị thông tin khách hàng"""
    print("\n" + "="*80)
    print("CUSTOMER SUMMARY")
    print("="*80)
    
    latest = customer_df.iloc[-1]
    
    print(f"Customer ID: {latest['MÃ KHÁCH HÀNG']}")
    print(f"Customer Name: {latest.get('TÊN KHÁCH HÀNG', 'N/A')}")
    print(f"Province: {latest.get('TỈNH/TP CỦA KHÁCH HÀNG', 'N/A')}")
    print(f"Total Orders: {len(customer_df):,}")
    print(f"Total Revenue: {customer_df['THÀNH TIỀN ĐƠN BÁN'].sum():,.0f} VND")
    print(f"Average Order: {customer_df['THÀNH TIỀN ĐƠN BÁN'].mean():,.0f} VND")
    
    print(f"\nLast Order Date: {latest['NGÀY TẠO ĐƠN BÁN']}")
    
    print("\nTop 5 Products Purchased:")
    top_products = customer_df['MÃ SẢN PHẨM ĐƠN BÁN'].value_counts().head(5)
    for i, (product, count) in enumerate(top_products.items(), 1):
        print(f"  {i}. {product}: {count} times")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Predict next purchase for customer')
    parser.add_argument('--customer-id', type=str, required=True, help='Customer ID')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--data-path', type=str, default='../data/raw/merged_2025.csv',
                       help='Path to data file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--max-rows', type=int, default=100000, 
                       help='Max rows to load from data')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔮 SALES PREDICTION SYSTEM")
    print("="*80)
    
    # Load config
    config = get_config('development')
    
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using: python minimal_train.py")
        sys.exit(1)
    
    model, product_to_id, id_to_product = load_model(model_path, config)
    
    # Load customer history
    data_path = Path(args.data_path)
    customer_df = get_customer_history(args.customer_id, data_path, args.max_rows)
    
    if customer_df is None:
        sys.exit(1)
    
    # Show customer summary
    show_customer_summary(customer_df)
    
    # Prepare features
    cache_file = config.CACHE_DIR / 'geocache.pkl'
    geo_cache = GeoCache(cache_file=cache_file)
    
    sequence = prepare_customer_features(customer_df, config, geo_cache)
    
    if sequence is None:
        sys.exit(1)
    
    # Predict
    predictions = predict_next_purchase(
        model, sequence, product_to_id, id_to_product, config, args.top_k
    )
    
    # Save predictions
    output_file = f"predictions_{args.customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'customer_id': args.customer_id,
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Predictions saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()