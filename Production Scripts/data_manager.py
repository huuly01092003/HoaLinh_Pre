"""
Data Manager for Processed Data
Handles saving, loading, and caching of processed features
"""
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    
import pickle
import gzip
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import hashlib

logger = logging.getLogger(__name__)


class DataManager:
    """Manage processed data lifecycle"""
    
    def __init__(self, processed_dir: Path, use_compression: bool = True):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.use_compression = use_compression
        self.ext = '.pkl.gz' if use_compression else '.pkl'
    
    def _get_file_path(self, name: str) -> Path:
        """Get file path with appropriate extension"""
        return self.processed_dir / f"{name}{self.ext}"
    
    def _save_pickle(self, data, path: Path):
        """Save data with optional compression"""
        if self.use_compression:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    
    def _load_pickle(self, path: Path):
        """Load data with optional compression"""
        if self.use_compression:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def save_sequences(
        self,
        train_sequences: pd.DataFrame,
        val_sequences: pd.DataFrame,
        test_sequences: pd.DataFrame,
        version: str = 'latest'
    ):
        """Save all sequence dataframes"""
        logger.info(f"Saving sequences (version: {version})...")
        
        datasets = {
            f'train_sequences_{version}': train_sequences,
            f'val_sequences_{version}': val_sequences,
            f'test_sequences_{version}': test_sequences
        }
        
        for name, data in datasets.items():
            path = self._get_file_path(name)
            self._save_pickle(data, path)
            size_mb = path.stat().st_size / 1024 / 1024
            logger.info(f"  ✓ Saved {name}: {len(data):,} rows, {size_mb:.1f} MB")
        
        # Save metadata
        self._save_metadata(version, train_sequences, val_sequences, test_sequences)
    
    def load_sequences(
        self,
        version: str = 'latest'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all sequence dataframes"""
        logger.info(f"Loading sequences (version: {version})...")
        
        train = self._load_pickle(self._get_file_path(f'train_sequences_{version}'))
        val = self._load_pickle(self._get_file_path(f'val_sequences_{version}'))
        test = self._load_pickle(self._get_file_path(f'test_sequences_{version}'))
        
        logger.info(f"  ✓ Train: {len(train):,} sequences")
        logger.info(f"  ✓ Val: {len(val):,} sequences")
        logger.info(f"  ✓ Test: {len(test):,} sequences")
        
        return train, val, test
    
    def save_engineered_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        version: str = 'latest'
    ):
        """Save engineered dataframes (before sequencing)"""
        logger.info(f"Saving engineered data (version: {version})...")
        
        datasets = {
            f'train_engineered_{version}': train_df,
            f'val_engineered_{version}': val_df,
            f'test_engineered_{version}': test_df
        }
        
        for name, data in datasets.items():
            path = self._get_file_path(name)
            self._save_pickle(data, path)
            size_mb = path.stat().st_size / 1024 / 1024
            logger.info(f"  ✓ Saved {name}: {len(data):,} rows, {size_mb:.1f} MB")
    
    def load_engineered_data(
        self,
        version: str = 'latest'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load engineered dataframes"""
        logger.info(f"Loading engineered data (version: {version})...")
        
        train = self._load_pickle(self._get_file_path(f'train_engineered_{version}'))
        val = self._load_pickle(self._get_file_path(f'val_engineered_{version}'))
        test = self._load_pickle(self._get_file_path(f'test_engineered_{version}'))
        
        logger.info(f"  ✓ Loaded {len(train):,} + {len(val):,} + {len(test):,} rows")
        
        return train, val, test
    
    def save_mappings(
        self,
        product_to_id: Dict,
        encoders: Dict,
        version: str = 'latest'
    ):
        """Save product mappings and encoders"""
        logger.info("Saving mappings and encoders...")
        
        self._save_pickle(product_to_id, self._get_file_path(f'product_mapping_{version}'))
        self._save_pickle(encoders, self._get_file_path(f'encoders_{version}'))
        
        logger.info(f"  ✓ Product mapping: {len(product_to_id):,} products")
        logger.info(f"  ✓ Encoders: {len(encoders)} encoders")
    
    def load_mappings(
        self,
        version: str = 'latest'
    ) -> Tuple[Dict, Dict]:
        """Load product mappings and encoders"""
        logger.info("Loading mappings and encoders...")
        
        product_to_id = self._load_pickle(self._get_file_path(f'product_mapping_{version}'))
        encoders = self._load_pickle(self._get_file_path(f'encoders_{version}'))
        
        logger.info(f"  ✓ Product mapping: {len(product_to_id):,} products")
        logger.info(f"  ✓ Encoders: {len(encoders)} encoders")
        
        return product_to_id, encoders
    
    def _save_metadata(
        self,
        version: str,
        train_sequences: pd.DataFrame,
        val_sequences: pd.DataFrame,
        test_sequences: pd.DataFrame
    ):
        """Save processing metadata"""
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'num_sequences': {
                'train': len(train_sequences),
                'val': len(val_sequences),
                'test': len(test_sequences),
                'total': len(train_sequences) + len(val_sequences) + len(test_sequences)
            },
            'num_features': len(train_sequences.columns),
            'compression': self.use_compression
        }
        
        metadata_path = self.processed_dir / f'metadata_{version}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  ✓ Metadata saved to {metadata_path}")
    
    def load_metadata(self, version: str = 'latest') -> Dict:
        """Load processing metadata"""
        metadata_path = self.processed_dir / f'metadata_{version}.json'
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def check_cache_validity(
        self,
        raw_data_path: Path,
        version: str = 'latest',
        max_age_days: int = 7
    ) -> bool:
        """Check if cached data is still valid"""
        
        # Check if files exist
        required_files = [
            f'train_sequences_{version}',
            f'val_sequences_{version}',
            f'test_sequences_{version}',
            f'product_mapping_{version}',
            f'encoders_{version}'
        ]
        
        for name in required_files:
            path = self._get_file_path(name)
            if not path.exists():
                logger.warning(f"Missing file: {path}")
                return False
        
        # Check age
        cache_path = self._get_file_path(f'train_sequences_{version}')
        cache_age_days = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 86400
        
        if cache_age_days > max_age_days:
            logger.warning(f"Cache too old: {cache_age_days:.1f} days")
            return False
        
        # Check if raw data is newer than cache
        if raw_data_path.exists():
            raw_mtime = raw_data_path.stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            
            if raw_mtime > cache_mtime:
                logger.warning("Raw data is newer than cache")
                return False
        
        logger.info(f"✓ Cache is valid (age: {cache_age_days:.1f} days)")
        return True
    
    def list_versions(self) -> list:
        """List all available versions"""
        versions = set()
        for file in self.processed_dir.glob(f'train_sequences_*{self.ext}'):
            version = file.stem.replace('train_sequences_', '').replace('.pkl', '')
            versions.add(version)
        
        return sorted(list(versions))
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached data"""
        total_size = 0
        file_count = 0
        
        for file in self.processed_dir.glob(f'*{self.ext}'):
            total_size += file.stat().st_size
            file_count += 1
        
        return {
            'total_size_mb': total_size / 1024 / 1024,
            'total_size_gb': total_size / 1024 / 1024 / 1024,
            'file_count': file_count,
            'versions': self.list_versions()
        }
    
    def clean_old_versions(self, keep_latest: int = 2):
        """Remove old versions, keep only latest N"""
        versions = self.list_versions()
        
        if len(versions) <= keep_latest:
            logger.info(f"No cleanup needed ({len(versions)} versions)")
            return
        
        versions_to_remove = versions[:-keep_latest]
        
        for version in versions_to_remove:
            logger.info(f"Removing version: {version}")
            
            # Remove all files for this version
            for file in self.processed_dir.glob(f'*_{version}*'):
                file.unlink()
                logger.info(f"  ✓ Removed {file.name}")
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for change detection"""
        # Use a sample of data for speed
        sample = df.head(1000).to_json()
        return hashlib.md5(sample.encode()).hexdigest()


def load_or_create_sequences(
    raw_data_path: Path,
    config,
    geo_cache,
    force_rebuild: bool = False,
    version: str = 'latest'
):
    """
    High-level function to load sequences from cache or create new ones
    
    Usage:
        train_seq, val_seq, test_seq, product_to_id, encoders = load_or_create_sequences(
            raw_data_path='data/raw/merged_2025.csv',
            config=config,
            geo_cache=geo_cache,
            force_rebuild=False
        )
    """
    from sklearn.model_selection import train_test_split
    from features import FeatureEngineer
    
    data_manager = DataManager(config.PROCESSED_DATA_PATH)
    
    # Check cache validity
    cache_valid = data_manager.check_cache_validity(
        raw_data_path=raw_data_path,
        version=version,
        max_age_days=7
    )
    
    if cache_valid and not force_rebuild:
        logger.info("📂 Loading from cache...")
        
        try:
            train_sequences, val_sequences, test_sequences = data_manager.load_sequences(version)
            product_to_id, encoders = data_manager.load_mappings(version)
            
            logger.info("✅ Successfully loaded from cache!")
            return train_sequences, val_sequences, test_sequences, product_to_id, encoders
        
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            logger.info("Falling back to rebuild...")
    
    # Create new processed data
    logger.info("🔧 Creating new processed data...")
    
    # Load raw data
    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_csv(config.RAW_DATA_PATH, encoding='utf-8-sig', low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.15, random_state=config.RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.7, random_state=config.RANDOM_STATE)
    
    # Feature engineering
    logger.info("Processing training set...")
    train_engineer = FeatureEngineer(train_df, config.CURRENT_DATE, geo_cache)
    train_engineer.process(fit_mode=True)
    
    logger.info("Processing validation set...")
    val_engineer = FeatureEngineer(val_df, config.CURRENT_DATE, geo_cache)
    val_engineer.encoders = train_engineer.encoders
    val_engineer.process(fit_mode=False)
    
    logger.info("Processing test set...")
    test_engineer = FeatureEngineer(test_df, config.CURRENT_DATE, geo_cache)
    test_engineer.encoders = train_engineer.encoders
    test_engineer.process(fit_mode=False)
    
    # Save engineered data (optional, for debugging)
    data_manager.save_engineered_data(
        train_engineer.df, val_engineer.df, test_engineer.df, version
    )
    
    # Create sequences
    logger.info("Creating sequences...")
    train_sequences = train_engineer.prepare_sequences(config.MAX_SEQ_LEN)
    val_sequences = val_engineer.prepare_sequences(config.MAX_SEQ_LEN)
    test_sequences = test_engineer.prepare_sequences(config.MAX_SEQ_LEN)
    
    # Product mapping
    unique_products = train_df['MÃ SẢN PHẨM ĐƠN BÁN'].unique()
    product_to_id = {p: i+1 for i, p in enumerate(unique_products)}
    
    # Save to cache
    logger.info("💾 Saving to cache...")
    data_manager.save_sequences(train_sequences, val_sequences, test_sequences, version)
    data_manager.save_mappings(product_to_id, train_engineer.encoders, version)
    
    # Print cache stats
    stats = data_manager.get_cache_stats()
    logger.info(f"📊 Cache statistics:")
    logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
    logger.info(f"  Files: {stats['file_count']}")
    logger.info(f"  Versions: {', '.join(stats['versions'])}")
    
    logger.info("✅ Processing complete!")
    
    return train_sequences, val_sequences, test_sequences, product_to_id, train_engineer.encoders