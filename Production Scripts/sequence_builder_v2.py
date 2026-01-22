"""
Sequence Builder V2 - ĐƠN ĐẶT HÀNG (NO MONEY)
Focus: Products + Quantities của đơn đặt
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class NormalizationStats:
    """Normalization statistics"""
    
    def __init__(self):
        self.num_skus_mean = 0
        self.num_skus_std = 1
        self.quantity_mean = 0
        self.quantity_std = 1
    
    def fit(self, orders_df):
        """Calculate stats from training data"""
        # Num SKUs per order
        self.num_skus_mean = orders_df['NUM_SKUS'].mean()
        self.num_skus_std = orders_df['NUM_SKUS'].std()
        
        # Total quantity per order
        self.quantity_mean = orders_df['TOTAL_QTY'].mean()
        self.quantity_std = orders_df['TOTAL_QTY'].std()
        
        logger.info(f"📊 Normalization Stats (ĐƠN ĐẶT):")
        logger.info(f"  Num SKUs: mean={self.num_skus_mean:.2f}, std={self.num_skus_std:.2f}")
        logger.info(f"  Quantity: mean={self.quantity_mean:.2f}, std={self.quantity_std:.2f}")
    
    def normalize_num_skus(self, value):
        return (value - self.num_skus_mean) / (self.num_skus_std + 1e-8)
    
    def denormalize_num_skus(self, value):
        return value * self.num_skus_std + self.num_skus_mean
    
    def normalize_quantity(self, value):
        return (value - self.quantity_mean) / (self.quantity_std + 1e-8)
    
    def denormalize_quantity(self, value):
        return value * self.quantity_std + self.quantity_mean
    
    def normalize_days(self, value):
        return np.log1p(value)
    
    def denormalize_days(self, value):
        return np.expm1(value)


class SequenceBuilderV2:
    """Xây dựng sequences - ĐƠN ĐẶT"""
    
    def __init__(self, config, norm_stats=None):
        self.cfg = config
        self.history_len = config.HISTORY_LEN
        self.norm_stats = norm_stats if norm_stats else NormalizationStats()
    
    def build_customer_sequence(self, customer_orders: pd.DataFrame) -> List[Dict]:
        """Build sequences cho 1 customer"""
        orders = customer_orders.sort_values('NGÀY TẠO ĐƠN ĐẶT').reset_index(drop=True)
        
        if len(orders) <= self.history_len:
            return []
        
        sequences = []
        
        for i in range(len(orders) - self.history_len):
            history = orders.iloc[i:i+self.history_len]
            target = orders.iloc[i+self.history_len]
            
            seq = self._create_sequence(history, target)
            sequences.append(seq)
        
        return sequences
    
    def _create_sequence(self, history: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Tạo 1 sequence - ĐƠN ĐẶT (NO MONEY)
        
        Logic:
        - hist_products: List of lists (mỗi order có nhiều products)
        - hist_quantities: List of lists (mỗi order có nhiều quantities)
        - hist_total_qty: List of floats (tổng số lượng mỗi order)
        - hist_num_skus: List of floats (số SKUs mỗi order)
        """
        
        # ===== HISTORY =====
        
        # 1. Products (List of lists)
        hist_products = history['MÃ SẢN PHẨM ĐƠN ĐẶT'].tolist()
        
        # 2. Quantities per SKU (List of lists)
        hist_quantities = history['SỐ LƯỢNG ĐƠN ĐẶT'].tolist()
        
        # 3. Total quantity per order (NORMALIZED)
        hist_total_qty_raw = history['TOTAL_QTY'].tolist()
        hist_total_qty = [
            self.norm_stats.normalize_quantity(x) for x in hist_total_qty_raw
        ]
        
        # 4. Num SKUs per order (NORMALIZED)
        hist_num_skus_raw = history['NUM_SKUS'].tolist()
        hist_num_skus = [
            self.norm_stats.normalize_num_skus(x) for x in hist_num_skus_raw
        ]
        
        # 5. Days between orders (NORMALIZED)
        hist_dates = history['NGÀY TẠO ĐƠN ĐẶT'].tolist()
        days_between = []
        for i in range(1, len(hist_dates)):
            delta = (hist_dates[i] - hist_dates[i-1]).days
            days_between.append(self.norm_stats.normalize_days(delta))
        days_between = [0] + days_between
        
        # ===== CUSTOMER INFO =====
        customer_id = history.iloc[0]['MÃ KHÁCH HÀNG']
        employee_id = history.iloc[0]['MÃ NHÂN VIÊN']
        route_id = history.iloc[0]['TUYẾN BÁN HÀNG']
        
        # ===== TARGET =====
        
        # 1. Products to predict
        target_products = target['MÃ SẢN PHẨM ĐƠN ĐẶT']
        
        # 2. Quantities per SKU
        target_quantities = target['SỐ LƯỢNG ĐƠN ĐẶT']
        
        # 3. Total quantity (NORMALIZED)
        target_total_qty = self.norm_stats.normalize_quantity(
            target['TOTAL_QTY']
        )
        
        # 4. Num SKUs (NORMALIZED)
        target_num_skus = self.norm_stats.normalize_num_skus(
            target['NUM_SKUS']
        )
        
        # 5. Days until next order (NORMALIZED)
        last_hist_date = history.iloc[-1]['NGÀY TẠO ĐƠN ĐẶT']
        target_date = target['NGÀY TẠO ĐƠN ĐẶT']
        days_until_next = self.norm_stats.normalize_days(
            (target_date - last_hist_date).days
        )
        
        return {
            # ===== HISTORY =====
            'hist_products': hist_products,
            'hist_quantities': hist_quantities,
            'hist_total_qty': hist_total_qty,
            'hist_num_skus': hist_num_skus,
            'hist_days_between': days_between,
            
            # ===== CUSTOMER INFO =====
            'customer_id': customer_id,
            'employee_id': employee_id,
            'route_id': route_id,
            
            # ===== TARGET =====
            'target_products': target_products,
            'target_quantities': target_quantities,
            'target_total_qty': target_total_qty,
            'target_num_skus': target_num_skus,
            'days_until_next': days_until_next,
            
            # ===== METADATA =====
            'last_hist_date': last_hist_date,
            'target_date': target_date
        }
    
    def build_all_sequences(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Build sequences cho tất cả customers"""
        logger.info(f"Building sequences (ĐƠN ĐẶT) for {orders_df['MÃ KHÁCH HÀNG'].nunique():,} customers...")
        
        all_sequences = []
        
        for customer_id, customer_orders in orders_df.groupby('MÃ KHÁCH HÀNG'):
            sequences = self.build_customer_sequence(customer_orders)
            all_sequences.extend(sequences)
        
        logger.info(f"✅ Created {len(all_sequences):,} sequences (ĐƠN ĐẶT - NO MONEY)")
        
        sequences_df = pd.DataFrame(all_sequences)
        
        return sequences_df


class VocabularyBuilder:
    """Build vocabulary cho products"""
    
    def __init__(self):
        self.product_vocab = {'<PAD>': 0, '<UNK>': 1}
        self.next_product_id = 2
    
    def fit(self, sequences_df: pd.DataFrame):
        """Fit vocabulary từ sequences"""
        logger.info("Building vocabulary (ĐƠN ĐẶT)...")
        
        all_products = set()
        for products_list in sequences_df['hist_products']:
            for products in products_list:
                if isinstance(products, list):
                    all_products.update(products)
                else:
                    all_products.add(products)
        
        for products_list in sequences_df['target_products']:
            if isinstance(products_list, list):
                all_products.update(products_list)
            else:
                all_products.add(products_list)
        
        for product in sorted(all_products):
            if product not in self.product_vocab and product != 'UNKNOWN':
                self.product_vocab[product] = self.next_product_id
                self.next_product_id += 1
        
        logger.info(f"✅ Product vocab size: {len(self.product_vocab):,}")
        
        return self
    
    def transform_product(self, product):
        return self.product_vocab.get(product, 1)
    
    def transform_products_list(self, products):
        if isinstance(products, list):
            return [self.transform_product(p) for p in products]
        return [self.transform_product(products)]
    
    def get_product_name(self, product_id):
        for name, pid in self.product_vocab.items():
            if pid == product_id:
                return name
        return '<UNK>'
    
    @property
    def product_vocab_size(self):
        return len(self.product_vocab)


if __name__ == "__main__":
    from config_v2 import SystemConfig
    import pickle
    
    logging.basicConfig(level=logging.INFO)
    SystemConfig.init_dirs()
    
    # Load orders
    train_orders = pd.read_pickle(SystemConfig.PROCESSED / 'train_orders.pkl')
    
    # Fit normalization
    norm_stats = NormalizationStats()
    norm_stats.fit(train_orders)
    
    # Build sequences
    builder = SequenceBuilderV2(SystemConfig, norm_stats)
    train_sequences = builder.build_all_sequences(train_orders)
    
    print(f"\n✅ Sequences created: {len(train_sequences):,}")
    print(f"\nColumns:")
    for col in train_sequences.columns:
        print(f"  - {col}")
    
    # Build vocabulary
    vocab = VocabularyBuilder()
    vocab.fit(train_sequences)
    
    print(f"\nProduct vocab size: {vocab.product_vocab_size:,}")
    
    # Save
    train_sequences.to_pickle(SystemConfig.PROCESSED / 'train_sequences.pkl')
    with open(SystemConfig.PROCESSED / 'vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(SystemConfig.PROCESSED / 'norm_stats.pkl', 'wb') as f:
        pickle.dump(norm_stats, f)
    
    print("\n✅ Test complete!")