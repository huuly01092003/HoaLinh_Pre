"""
Feature Engineering Module for Sales Prediction
Comprehensive feature extraction and transformation
"""
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict
import logging

from geocache import GeoCache

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self, df: pd.DataFrame, current_date: datetime, 
                 geo_cache: Optional[GeoCache] = None):
        self.df = df.copy()
        self.current_date = current_date
        self.geo_cache = geo_cache or GeoCache()
        self.encoders = {}
        self.scalers = {}
    
    def create_temporal_features(self):
        """Extract time-based features"""
        logger.info("Creating temporal features...")
        
        self.df['NGÀY TẠO ĐƠN BÁN'] = pd.to_datetime(
            self.df['NGÀY TẠO ĐƠN BÁN'], format='%d/%m/%Y', errors='coerce'
        )
        
        # Time components
        self.df['year'] = self.df['NGÀY TẠO ĐƠN BÁN'].dt.year
        self.df['month'] = self.df['NGÀY TẠO ĐƠN BÁN'].dt.month
        self.df['quarter'] = self.df['NGÀY TẠO ĐƠN BÁN'].dt.quarter
        self.df['week'] = self.df['NGÀY TẠO ĐƠN BÁN'].dt.isocalendar().week
        self.df['day_of_week'] = self.df['NGÀY TẠO ĐƠN BÁN'].dt.dayofweek
        self.df['day_of_month'] = self.df['NGÀY TẠO ĐƠN BÁN'].dt.day
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_month_start'] = (self.df['day_of_month'] <= 7).astype(int)
        self.df['is_month_end'] = (self.df['day_of_month'] >= 23).astype(int)
        
        # Hour features
        self.df['THỜI GIAN TẠO ĐƠN BÁN'] = pd.to_datetime(
            self.df['THỜI GIAN TẠO ĐƠN BÁN'], format='%H:%M', errors='coerce'
        )
        self.df['hour'] = self.df['THỜI GIAN TẠO ĐƠN BÁN'].dt.hour.fillna(0).astype(int)
        self.df['is_morning'] = ((self.df['hour'] >= 6) & (self.df['hour'] < 12)).astype(int)
        self.df['is_afternoon'] = ((self.df['hour'] >= 12) & (self.df['hour'] < 18)).astype(int)
        self.df['is_evening'] = (self.df['hour'] >= 18).astype(int)
        
        # Recency
        self.df['recency_days'] = (
            self.current_date - self.df['NGÀY TẠO ĐƠN BÁN']
        ).dt.days.fillna(9999).astype(int)
        
        return self
    
    def clean_numeric_columns(self):
        """Clean numeric columns"""
        logger.info("Cleaning numeric columns...")
        
        numeric_columns = [
            'SỐ LƯỢNG ĐƠN BÁN', 'ĐƠN GIÁ ĐƠN BÁN', 'TỔNG TIỀN ĐƠN BÁN',
            'CK SP ĐƠN BÁN', 'CK ĐƠN HÀNG ĐƠN BÁN', 'TỔNG CHIẾT KHẤU ĐƠN BÁN',
            'CHIẾT KHẤU ĐƠN BÁN', 'THÀNH TIỀN ĐƠN BÁN'
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        return self
    
    def clean_text_columns(self):
        """Clean text columns"""
        logger.info("Cleaning text columns...")
        
        text_columns = [
            'MÃ KHÁCH HÀNG', 'TÊN KHÁCH HÀNG', 'MÃ NHÂN VIÊN',
            'TUYẾN BÁN HÀNG', 'TỈNH/TP CỦA KHÁCH HÀNG',
            'QUẬN/HUYỆN CỦA KHÁCH HÀNG', 'MÃ SẢN PHẨM ĐƠN BÁN',
            'LOẠI SẢN PHẨM', 'NGUỒN ĐƠN'
        ]
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.df[col].replace(['nan', 'None', '0', 'N/A'], '')
                self.df[col] = self.df[col].str.strip()
        
        return self
    
    def create_product_features(self):
        """Extract product features"""
        logger.info("Creating product features...")
        
        self.df['MÃ SẢN PHẨM ĐƠN BÁN'] = self.df['MÃ SẢN PHẨM ĐƠN BÁN'].replace('', 'UNKNOWN')
        self.df['product_series'] = self.df['MÃ SẢN PHẨM ĐƠN BÁN'].str[:4]
        self.df['product_family'] = self.df['MÃ SẢN PHẨM ĐƠN BÁN'].str[:2]
        
        self.df['unit_price'] = self.df['ĐƠN GIÁ ĐƠN BÁN'].copy()
        
        try:
            self.df['price_tier'] = pd.qcut(
                self.df['unit_price'], q=5,
                labels=['very_low', 'low', 'medium', 'high', 'premium'],
                duplicates='drop'
            )
        except:
            price_bins = [0, 10000, 30000, 60000, 100000, float('inf')]
            self.df['price_tier'] = pd.cut(
                self.df['unit_price'], bins=price_bins,
                labels=['very_low', 'low', 'medium', 'high', 'premium'],
                include_lowest=True
            )
        
        self.df['discount_rate'] = (
            self.df['CHIẾT KHẤU ĐƠN BÁN'] / 
            self.df['TỔNG TIỀN ĐƠN BÁN'].replace(0, 1)
        ).clip(0, 1)
        
        self.df['has_discount'] = (self.df['CHIẾT KHẤU ĐƠN BÁN'] > 0).astype(int)
        self.df['order_value'] = self.df['THÀNH TIỀN ĐƠN BÁN'].copy()
        
        return self
    
    def create_customer_features(self):
        """Create customer behavior features"""
        logger.info("Creating customer features...")
        
        self.df['is_walkin'] = (
            self.df.get('LÀ KHÁCH HÀNG VÃNG LAI', '') == 'Có'
        ).astype(int)
        
        self.df['is_mobile'] = (self.df['NGUỒN ĐƠN'] == 'Mobile').astype(int)
        self.df['is_web'] = (self.df['NGUỒN ĐƠN'] == 'Web').astype(int)
        
        return self
    
    def create_route_features(self):
        """Create geography and route features"""
        logger.info("Creating route and geography features...")
        
        self.df['route_code'] = self.df['TUYẾN BÁN HÀNG'].replace('', 'NO_ROUTE')
        self.df['province'] = self.df['TỈNH/TP CỦA KHÁCH HÀNG'].replace('', 'Unknown')
        self.df['district'] = self.df['QUẬN/HUYỆN CỦA KHÁCH HÀNG'].replace('', 'Unknown')
        
        # Geocode unique addresses
        unique_addresses = self.df[[
            'ĐỊA CHỈ CỦA KHÁCH HÀNG', 'QUẬN/HUYỆN CỦA KHÁCH HÀNG',
            'TỈNH/TP CỦA KHÁCH HÀNG'
        ]].drop_duplicates()
        
        logger.info(f"Geocoding {len(unique_addresses):,} unique addresses...")
        coords_list = self.geo_cache.batch_geocode(unique_addresses)
        unique_addresses['lat'] = [c[0] for c in coords_list]
        unique_addresses['lon'] = [c[1] for c in coords_list]
        
        self.df = self.df.merge(unique_addresses, 
            on=['ĐỊA CHỈ CỦA KHÁCH HÀNG', 'QUẬN/HUYỆN CỦA KHÁCH HÀNG', 
                'TỈNH/TP CỦA KHÁCH HÀNG'], how='left')
        
        # Fill missing coordinates
        missing = self.df['lat'].isna()
        if missing.sum() > 0:
            province_coords = self.df.loc[missing, 'province'].apply(
                lambda p: self.geo_cache.get_province_center(p)
            )
            self.df.loc[missing, 'lat'] = province_coords.apply(lambda x: x[0])
            self.df.loc[missing, 'lon'] = province_coords.apply(lambda x: x[1])
        
        # Calculate distances
        self._calculate_distances()
        
        return self
    
    def _calculate_distances(self):
        """Calculate geographic distances"""
        valid = (self.df['lat'].notna()) & (self.df['lon'].notna())
        
        # Employee base locations
        emp_bases = self.df[valid].groupby('MÃ NHÂN VIÊN').agg({
            'lat': 'median', 'lon': 'median'
        }).reset_index()
        emp_bases.columns = ['MÃ NHÂN VIÊN', 'emp_lat', 'emp_lon']
        
        self.df = self.df.merge(emp_bases, on='MÃ NHÂN VIÊN', how='left')
        
        self.df['distance_to_employee'] = self.geo_cache.haversine(
            self.df['lat'].fillna(0).values,
            self.df['lon'].fillna(0).values,
            self.df['emp_lat'].fillna(self.df['lat'].fillna(0)).values,
            self.df['emp_lon'].fillna(self.df['lon'].fillna(0)).values
        )
        
        # Major cities
        self.df['distance_to_hcmc'] = self.geo_cache.haversine(
            self.df['lat'].fillna(0), self.df['lon'].fillna(0), 10.8231, 106.6297
        )
        self.df['distance_to_hanoi'] = self.geo_cache.haversine(
            self.df['lat'].fillna(0), self.df['lon'].fillna(0), 21.0285, 105.8542
        )
        
        self.df['is_urban'] = (
            (self.df['distance_to_hcmc'] < 30) | (self.df['distance_to_hanoi'] < 30)
        ).astype(int)
    
    def create_aggregated_features(self):
        """Create customer-level aggregated features"""
        logger.info("Creating aggregated features...")
        
        self.df = self.df.sort_values(['MÃ KHÁCH HÀNG', 'NGÀY TẠO ĐƠN BÁN'])
        
        # Customer stats
        customer_stats = self.df.groupby('MÃ KHÁCH HÀNG').agg({
            'MÃ ĐƠN BÁN HÀNG': 'count',
            'THÀNH TIỀN ĐƠN BÁN': ['sum', 'mean', 'std'],
            'SỐ LƯỢNG ĐƠN BÁN': ['sum', 'mean'],
            'CHIẾT KHẤU ĐƠN BÁN': 'mean',
            'NGÀY TẠO ĐƠN BÁN': ['min', 'max']
        }).reset_index()
        
        # FIX: Flatten columns properly
        # MultiIndex columns look like: ('MÃ KHÁCH HÀNG', '') or ('THÀNH TIỀN ĐƠN BÁN', 'mean')
        customer_stats.columns = [
            '_'.join(col).strip('_') if col[1] else col[0] 
            for col in customer_stats.columns.values
        ]
        
        # Print for debug
        print(f"DEBUG: Columns after proper flatten:")
        print(customer_stats.columns.tolist())
        
        # RFM - Sử dụng tên đã flatten
        customer_stats['frequency'] = customer_stats['MÃ ĐƠN BÁN HÀNG_count']
        customer_stats['monetary'] = customer_stats['THÀNH TIỀN ĐƠN BÁN_mean']
        
        # Recency
        customer_stats['recency'] = (
            self.current_date - pd.to_datetime(customer_stats['NGÀY TẠO ĐƠN BÁN_max'])
        ).dt.days.fillna(9999).astype(int)
        
        # Customer lifetime
        customer_stats['customer_lifetime'] = (
            pd.to_datetime(customer_stats['NGÀY TẠO ĐƠN BÁN_max']) -
            pd.to_datetime(customer_stats['NGÀY TẠO ĐƠN BÁN_min'])
        ).dt.days.fillna(0).astype(int)
        
        # Merge back
        self.df = self.df.merge(customer_stats, on='MÃ KHÁCH HÀNG', how='left')
        
        # Product diversity
        prod_div = self.df.groupby('MÃ KHÁCH HÀNG').agg({
            'MÃ SẢN PHẨM ĐƠN BÁN': 'nunique',
            'LOẠI SẢN PHẨM': 'nunique'
        }).reset_index()
        prod_div.columns = ['MÃ KHÁCH HÀNG', 'num_unique_products', 'num_categories']
        
        self.df = self.df.merge(prod_div, on='MÃ KHÁCH HÀNG', how='left')
        
        # Temporal comparisons
        self.df = self._create_temporal_comparisons()
        
        return self
    
    def _create_temporal_comparisons(self):
        """Create period-over-period comparisons"""
        logger.info("Creating temporal comparison features...")
        
        df_list = []
        
        for customer_id, group in tqdm(self.df.groupby('MÃ KHÁCH HÀNG'), 
                                       desc="Temporal comparisons"):
            group = group.sort_values('NGÀY TẠO ĐƠN BÁN').reset_index(drop=True)
            
            # Weekly
            group['week_year'] = group['year'] * 100 + group['week']
            weekly = group.groupby('week_year')['THÀNH TIỀN ĐƠN BÁN'].agg(['sum', 'count'])
            weekly['week_revenue_change'] = weekly['sum'].diff()
            weekly['week_order_change'] = weekly['count'].diff()
            group = group.merge(weekly[['week_revenue_change', 'week_order_change']], 
                               left_on='week_year', right_index=True, how='left')
            
            # Monthly
            group['month_year'] = group['year'] * 100 + group['month']
            monthly = group.groupby('month_year')['THÀNH TIỀN ĐƠN BÁN'].agg(['sum', 'count'])
            monthly['month_revenue_change'] = monthly['sum'].diff()
            monthly['month_order_change'] = monthly['count'].diff()
            group = group.merge(monthly[['month_revenue_change', 'month_order_change']],
                               left_on='month_year', right_index=True, how='left')
            
            # Quarterly
            group['quarter_year'] = group['year'] * 10 + group['quarter']
            quarterly = group.groupby('quarter_year')['THÀNH TIỀN ĐƠN BÁN'].agg(['sum', 'count'])
            quarterly['quarter_revenue_change'] = quarterly['sum'].diff()
            quarterly['quarter_order_change'] = quarterly['count'].diff()
            group = group.merge(quarterly[['quarter_revenue_change', 'quarter_order_change']],
                               left_on='quarter_year', right_index=True, how='left')
            
            # Yearly
            yearly = group.groupby('year')['THÀNH TIỀN ĐƠN BÁN'].agg(['sum', 'count'])
            yearly['year_revenue_change'] = yearly['sum'].diff()
            yearly['year_order_change'] = yearly['count'].diff()
            group = group.merge(yearly[['year_revenue_change', 'year_order_change']],
                               left_on='year', right_index=True, how='left')
            
            df_list.append(group)
        
        return pd.concat(df_list, ignore_index=True)
    
    def segment_customers(self, n_clusters: int = 5):
        """Cluster customers using RFM"""
        logger.info(f"Segmenting customers into {n_clusters} clusters...")
        
        rfm = self.df[['recency', 'frequency', 'monetary']].fillna(0)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['customer_segment'] = kmeans.fit_predict(rfm_scaled)
        
        return self
    
    def encode_categorical(self, fit_mode: bool = True):
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        cat_cols = [
            'LOẠI SẢN PHẨM', 'NGUỒN ĐƠN', 'product_series',
            'product_family', 'price_tier', 'province', 'district', 'route_code'
        ]
        
        for col in cat_cols:
            if col in self.df.columns:
                if fit_mode:
                    self.encoders[col] = LabelEncoder()
                    self.df[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        self.df[col].astype(str).fillna('Unknown')
                    )
                else:
                    self.df[f'{col}_encoded'] = self.encoders[col].transform(
                        self.df[col].astype(str).fillna('Unknown')
                    )
        
        return self
    
    def prepare_sequences(self, max_seq_len: int = 30) -> pd.DataFrame:
        """Create sequences for each customer"""
        logger.info("Preparing customer sequences...")
        
        sequences = []
        
        for customer_id, group in tqdm(self.df.groupby('MÃ KHÁCH HÀNG'), 
                                       desc="Creating sequences"):
            group = group.sort_values('NGÀY TẠO ĐƠN BÁN').reset_index(drop=True)
            
            if len(group) == 0:
                continue
            
            # Sequences
            product_seq = group['MÃ SẢN PHẨM ĐƠN BÁN'].tolist()[-max_seq_len:]
            qty_seq = group['SỐ LƯỢNG ĐƠN BÁN'].fillna(0).tolist()[-max_seq_len:]
            revenue_seq = group['order_value'].fillna(0).tolist()[-max_seq_len:]
            discount_seq = group['discount_rate'].fillna(0).tolist()[-max_seq_len:]
            
            week_change = group['week_revenue_change'].fillna(0).tolist()[-max_seq_len:]
            month_change = group['month_revenue_change'].fillna(0).tolist()[-max_seq_len:]
            quarter_change = group['quarter_revenue_change'].fillna(0).tolist()[-max_seq_len:]
            year_change = group['year_revenue_change'].fillna(0).tolist()[-max_seq_len:]
            
            latest = group.iloc[-1]
            
            sequences.append({
                'customer_id': customer_id,
                'product_seq': product_seq,
                'qty_seq': qty_seq,
                'revenue_seq': revenue_seq,
                'discount_seq': discount_seq,
                'week_change_seq': week_change,
                'month_change_seq': month_change,
                'quarter_change_seq': quarter_change,
                'year_change_seq': year_change,
                'recency': latest['recency'],
                'frequency': latest['frequency'],
                'monetary': latest['monetary'],
                'customer_lifetime': latest['customer_lifetime'],
                'num_unique_products': latest['num_unique_products'],
                'avg_discount': latest.get('CHIẾT KHẤU ĐƠN BÁN_mean', 0),
                'distance_to_employee': latest['distance_to_employee'],
                'customer_segment': latest['customer_segment'],
                'route_code': latest['route_code'],
                'employee_code': latest['MÃ NHÂN VIÊN'],
                'is_walkin': latest['is_walkin'],
                'is_weekend': latest['is_weekend'],
                'hour': latest['hour'],
                'day_of_week': latest['day_of_week'],
                'month': latest['month'],
                'quarter': latest['quarter'],
                'next_product': product_seq[-1] if product_seq else 'UNKNOWN',
                'next_quantity': qty_seq[-1] if qty_seq else 0,
                'next_revenue': revenue_seq[-1] if revenue_seq else 0,
                'next_discount': discount_seq[-1] if discount_seq else 0,
            })
        
        return pd.DataFrame(sequences)
    
    def process(self, fit_mode: bool = True):
        """Run full feature engineering pipeline"""
        logger.info("Running feature engineering pipeline...")
        
        self.clean_text_columns()
        self.clean_numeric_columns()
        self.create_temporal_features()
        self.create_product_features()
        self.create_customer_features()
        self.create_route_features()
        self.create_aggregated_features()
        self.segment_customers()
        self.encode_categorical(fit_mode=fit_mode)
        
        logger.info("Feature engineering complete!")
        return self
    
    # Trong features.py
    def process_in_chunks(self, chunk_size=50000):
        """Process data in chunks to avoid memory issues"""
        chunks = []
        for i in range(0, len(self.df), chunk_size):
            chunk = self.df.iloc[i:i+chunk_size].copy()
            # Process chunk
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    
    # Thêm vào features.py
def validate_data(self):
    """Validate data before processing"""
    required_columns = [
        'MÃ KHÁCH HÀNG', 'NGÀY TẠO ĐƠN BÁN', 
        'MÃ SẢN PHẨM ĐƠN BÁN', 'SỐ LƯỢNG ĐƠN BÁN'
    ]
    
    missing = [col for col in required_columns if col not in self.df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check minimum data requirements
    if len(self.df) < 1000:
        raise ValueError("Dataset too small for meaningful training")