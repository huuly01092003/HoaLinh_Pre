"""
Configuration Management for Sales Prediction System
FIXED VERSION - Correct path to DONHANG/data/
"""

from datetime import datetime
import torch
from pathlib import Path
import os

# ============================================================================
# WINDOWS + INTEL OPTIMIZATION
# ============================================================================

torch.set_num_threads(16)
torch.set_num_interop_threads(4)
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
torch.set_float32_matmul_precision('high')
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("="*80)
print("🚀 CPU OPTIMIZATION ACTIVE")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"MKL enabled: {torch.backends.mkl.is_available()}")
print(f"Platform: Windows + Intel Ultra 7 255H")
print("="*80)

# ============================================================================


class Config:
    """Base configuration class - FIXED PATHS"""
    
    # Project Structure
    PROJECT_ROOT = Path(__file__).parent  # Production Scripts/
    
    # ✅ FIX: Point to DONHANG/data/ (parent directory)
    DATA_DIR = PROJECT_ROOT.parent / 'data'
    
    MODEL_DIR = PROJECT_ROOT / 'models'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    CACHE_DIR = PROJECT_ROOT / 'cache'
    
    # Data Paths
    RAW_DATA_PATH = DATA_DIR / 'raw' / 'merged_2025.csv'
    PROCESSED_DATA_PATH = DATA_DIR / 'processed'
    
    # Model Paths
    BEST_MODEL_PATH = MODEL_DIR / 'best_sales_model.pth'
    CHECKPOINT_DIR = MODEL_DIR / 'checkpoints'
    
    # Training Parameters
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    PATIENCE = 5
    
    # Model Architecture
    EMBED_DIM = 128
    HIDDEN_SIZE = 256
    NUM_HEADS = 8
    DROPOUT = 0.3
    MAX_SEQ_LEN = 30
    NUM_CUSTOMER_SEGMENTS = 5
    
    # System
    DEVICE = torch.device("cpu")
    NUM_WORKERS = 0
    RANDOM_STATE = 42
    
    # Performance
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = 2
    
    # Business Logic
    CURRENT_DATE = datetime(2026, 1, 16)
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOGS_DIR, 
                         cls.CACHE_DIR, cls.PROCESSED_DATA_PATH, cls.CHECKPOINT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            key: getattr(cls, key) 
            for key in dir(cls) 
            if not key.startswith('_') and not callable(getattr(cls, key))
        }


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    BATCH_SIZE = 96
    NUM_WORKERS = 0
    LOG_LEVEL = 'WARNING'


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    LOG_LEVEL = 'DEBUG'


class ExperimentConfig(Config):
    """Experimentation configuration"""
    DEBUG = True
    BATCH_SIZE = 48
    EPOCHS = 50
    LEARNING_RATE = 0.0005
    LOG_LEVEL = 'INFO'
    EXPERIMENT_NAME = 'baseline'
    TRACK_EXPERIMENTS = True
    MLFLOW_TRACKING_URI = './mlruns'


class FastTestConfig(Config):
    """Fast testing configuration"""
    DEBUG = True
    BATCH_SIZE = 96
    EPOCHS = 3
    NUM_WORKERS = 0
    LOG_LEVEL = 'INFO'
    MAX_SEQ_LEN = 20


def get_config(env='development'):
    """Get configuration based on environment"""
    configs = {
        'production': ProductionConfig,
        'development': DevelopmentConfig,
        'experiment': ExperimentConfig,
        'fast': FastTestConfig
    }
    config = configs.get(env, DevelopmentConfig)
    
    print(f"\n📋 Config loaded: {env}", flush=True)
    print(f"   Batch size: {config.BATCH_SIZE}", flush=True)
    print(f"   Workers: {config.NUM_WORKERS}", flush=True)
    print(f"   Device: {config.DEVICE}", flush=True)
    print(f"   DATA_DIR: {config.DATA_DIR}", flush=True)
    print(f"   RAW_DATA_PATH: {config.RAW_DATA_PATH}", flush=True)
    print(f"   File exists: {config.RAW_DATA_PATH.exists()}", flush=True)
    print(flush=True)
    
    return config


if __name__ == "__main__":
    print("Testing configuration...")
    for env in ['development', 'production', 'fast']:
        cfg = get_config(env)
        print(f"✅ {env}: batch_size={cfg.BATCH_SIZE}, data_exists={cfg.RAW_DATA_PATH.exists()}")