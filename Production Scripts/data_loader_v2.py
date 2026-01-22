"""
Data Loader V2 - ĐƠN ĐẶT HÀNG (NO MONEY)
Focus: Products + Quantities của đơn đặt
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataLoaderV2:
    """Load và chuẩn bị dữ liệu - ĐƠN ĐẶT"""
    
    def __init__(self, config):
        self.cfg = config
    
    def load_csv(self):
        """Load CSV - CHỈ CÁC CỘT ĐƠN ĐẶT"""
        logger.info(f"Loading CSV: {self.cfg.RAW_CSV}")
        logger.info(f"Target: ĐƠN ĐẶT HÀNG (Purchase Orders)")
        logger.info(f"Columns to load: {len(self.cfg.REQUIRED_COLS)}")
        
        chunks = []
        for chunk in pd.read_csv(
            self.cfg.RAW_CSV,
            encoding='utf-8-sig',
            usecols=self.cfg.REQUIRED_COLS,
            chunksize=50000,
            low_memory=False
        ):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"✅ Loaded: {len(df):,} rows (ĐƠN ĐẶT)")
        return df
    
    def clean_data(self, df):
        """Làm sạch dữ liệu - ĐƠN ĐẶT (NO MONEY)"""
        logger.info("Cleaning data (ĐƠN ĐẶT - NO MONEY)...")
        
        # 1. Numeric columns - CHỈ QUANTITIES
        numeric_cols = [
            'SỐ LƯỢNG ĐƠN ĐẶT',         # Số lượng sản phẩm mỗi chi tiết
            'SỐ LƯỢNG SKUS ĐƠN ĐẶT'     # Số chi tiết đơn hàng
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            ).fillna(0)
            
            if col == 'SỐ LƯỢNG ĐƠN ĐẶT':
                logger.info(f"  {col}: sum = {df[col].sum():,.0f} (tổng số lượng sản phẩm)")
            else:
                logger.info(f"  {col}: sum = {df[col].sum():,.0f} (tổng số chi tiết đơn)")
        
        # 2. Text columns
        text_cols = [
            'MÃ KHÁCH HÀNG', 'MÃ NHÂN VIÊN', 'TUYẾN BÁN HÀNG',
            'MÃ SẢN PHẨM ĐƠN ĐẶT', 'LOẠI SẢN PHẨM', 'ĐƠN VỊ TÍNH ĐƠN ĐẶT'
        ]
        
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', ''], 'UNKNOWN')
        
        # 3. Datetime - ĐƠN ĐẶT
        df['NGÀY TẠO ĐƠN ĐẶT'] = pd.to_datetime(
            df['NGÀY TẠO ĐƠN ĐẶT'], 
            format='%d/%m/%Y', 
            errors='coerce'
        )
        df = df.dropna(subset=['NGÀY TẠO ĐƠN ĐẶT'])
        
        # 4. Filter: Chỉ giữ đơn hàng hợp lệ
        df = df[df['SỐ LƯỢNG ĐƠN ĐẶT'] > 0]
        
        logger.info(f"✅ Clean data: {len(df):,} rows (ĐƠN ĐẶT - NO MONEY)")
        return df
    
    def time_split(self, df):
        """
        Chia theo thời gian - BETTER BALANCE
        Train: 60% | Val: 20% | Test: 20%
        """
        logger.info("\n" + "="*80)
        logger.info("TIME-BASED SPLIT (ĐƠN ĐẶT)")
        logger.info("="*80)
        
        df = df.sort_values('NGÀY TẠO ĐƠN ĐẶT').reset_index(drop=True)
        
        n = len(df)
        train_idx = int(n * self.cfg.TRAIN_SPLIT)
        val_idx = int(n * (self.cfg.TRAIN_SPLIT + self.cfg.VAL_SPLIT))
        
        train = df.iloc[:train_idx].copy()
        val = df.iloc[train_idx:val_idx].copy()
        test = df.iloc[val_idx:].copy()
        
        # Log
        logger.info(f"Train: {train['NGÀY TẠO ĐƠN ĐẶT'].min().date()} → {train['NGÀY TẠO ĐƠN ĐẶT'].max().date()}")
        logger.info(f"  Rows: {len(train):,} | Customers: {train['MÃ KHÁCH HÀNG'].nunique():,}")
        
        logger.info(f"Val: {val['NGÀY TẠO ĐƠN ĐẶT'].min().date()} → {val['NGÀY TẠO ĐƠN ĐẶT'].max().date()}")
        logger.info(f"  Rows: {len(val):,} | Customers: {val['MÃ KHÁCH HÀNG'].nunique():,}")
        
        logger.info(f"Test: {test['NGÀY TẠO ĐƠN ĐẶT'].min().date()} → {test['NGÀY TẠO ĐƠN ĐẶT'].max().date()}")
        logger.info(f"  Rows: {len(test):,} | Customers: {test['MÃ KHÁCH HÀNG'].nunique():,}")
        logger.info("="*80 + "\n")
        
        return train, val, test
    
    def aggregate_to_orders(self, df):
        """
        Aggregate từ chi tiết đơn lên đơn hàng - ĐƠN ĐẶT
        
        Logic:
        - 1 MÃ ĐƠN ĐẶT HÀNG có nhiều chi tiết (SKUs)
        - SỐ LƯỢNG SKUS ĐƠN ĐẶT: Giống nhau cho tất cả chi tiết (số dòng)
        - SỐ LƯỢNG ĐƠN ĐẶT: Khác nhau từng chi tiết (số lượng sản phẩm)
        """
        logger.info("Aggregating to order level (ĐƠN ĐẶT)...")
        
        # Group by order
        orders = df.groupby([
            'MÃ ĐƠN ĐẶT HÀNG', 
            'MÃ KHÁCH HÀNG', 
            'MÃ NHÂN VIÊN', 
            'TUYẾN BÁN HÀNG', 
            'NGÀY TẠO ĐƠN ĐẶT'
        ]).agg({
            'MÃ SẢN PHẨM ĐƠN ĐẶT': lambda x: list(x),
            'SỐ LƯỢNG ĐƠN ĐẶT': lambda x: list(x),      # List quantities
            'ĐƠN VỊ TÍNH ĐƠN ĐẶT': lambda x: list(x),
            'SỐ LƯỢNG SKUS ĐƠN ĐẶT': 'first',            # Same for all lines
        }).reset_index()
        
        # Calculate metrics
        orders['NUM_SKUS'] = orders['MÃ SẢN PHẨM ĐƠN ĐẶT'].apply(len)
        orders['TOTAL_QTY'] = orders['SỐ LƯỢNG ĐƠN ĐẶT'].apply(sum)
        
        logger.info(f"✅ Orders: {len(orders):,}")
        logger.info(f"  Avg SKUs per order: {orders['NUM_SKUS'].mean():.2f}")
        logger.info(f"  Avg SKUs (from column): {orders['SỐ LƯỢNG SKUS ĐƠN ĐẶT'].mean():.2f}")
        logger.info(f"  Avg total quantity: {orders['TOTAL_QTY'].mean():.2f}")
        
        # Verify logic
        if abs(orders['NUM_SKUS'].mean() - orders['SỐ LƯỢNG SKUS ĐƠN ĐẶT'].mean()) > 0.1:
            logger.warning("⚠️ NUM_SKUS không khớp với SỐ LƯỢNG SKUS ĐƠN ĐẶT!")
        
        return orders
    
    def prepare_pipeline(self):
        """Pipeline hoàn chỉnh - ĐƠN ĐẶT"""
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION - ĐƠN ĐẶT HÀNG (NO MONEY)")
        logger.info("="*80 + "\n")
        
        # 1. Load
        df = self.load_csv()
        
        # 2. Clean
        df = self.clean_data(df)
        
        # 3. Split
        train_df, val_df, test_df = self.time_split(df)
        
        # 4. Aggregate
        train_orders = self.aggregate_to_orders(train_df)
        val_orders = self.aggregate_to_orders(val_df)
        test_orders = self.aggregate_to_orders(test_df)
        
        # 5. Save
        logger.info("Saving processed data...")
        train_orders.to_pickle(self.cfg.PROCESSED / 'train_orders.pkl')
        val_orders.to_pickle(self.cfg.PROCESSED / 'val_orders.pkl')
        test_orders.to_pickle(self.cfg.PROCESSED / 'test_orders.pkl')
        
        logger.info("\n✅ DATA PREPARATION COMPLETE (ĐƠN ĐẶT - NO MONEY)!")
        
        return train_orders, val_orders, test_orders


if __name__ == "__main__":
    from config_v2 import SystemConfig
    
    logging.basicConfig(level=logging.INFO)
    SystemConfig.init_dirs()
    
    loader = DataLoaderV2(SystemConfig)
    train, val, test = loader.prepare_pipeline()
    
    print(f"\n✅ Test complete!")
    print(f"Train orders: {len(train):,}")
    print(f"\nSample order:")
    print(train.iloc[0])
    print(f"\nColumns:")
    for col in train.columns:
        print(f"  - {col}")