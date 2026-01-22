"""
Configuration cho hệ thống dự đoán hành vi đặt hàng
FIXED: 
- Đổi ĐƠN BÁN → ĐƠN ĐẶT (đúng nghiệp vụ)
- Bỏ hoàn toàn các cột tiền
- Giảm overfitting
"""

import torch
from pathlib import Path
from datetime import datetime

class SystemConfig:
    """Cấu hình hệ thống - ĐƠN ĐẶT HÀNG"""
    
    # Paths
    ROOT = Path(__file__).parent
    DATA_DIR = ROOT.parent / 'data'
    RAW_CSV = DATA_DIR / 'raw' / 'merged_2025.csv'
    PROCESSED = DATA_DIR / 'processed_v2'
    MODELS = ROOT.parent / 'models_v2'
    LOGS = ROOT.parent / 'logs_v2'
    
    # Data columns - ĐƠN ĐẶT (NO MONEY)
    REQUIRED_COLS = [
        # Customer & Employee
        'MÃ KHÁCH HÀNG',
        'MÃ NHÂN VIÊN',
        'TUYẾN BÁN HÀNG',
        
        # Location (có thể dùng sau)
        'TỈNH/TP CỦA KHÁCH HÀNG',
        'QUẬN/HUYỆN CỦA KHÁCH HÀNG',
        
        # Transaction info - ĐƠN ĐẶT
        'NGÀY TẠO ĐƠN ĐẶT',
        'MÃ ĐƠN ĐẶT HÀNG',
        
        # Product details - CHỈ BEHAVIOR
        'MÃ SẢN PHẨM ĐƠN ĐẶT',
        'TÊN SẢN PHẨM ĐƠN ĐẶT',
        'LOẠI SẢN PHẨM',
        'ĐƠN VỊ TÍNH ĐƠN ĐẶT',
        
        # Quantities ONLY - NO MONEY
        'SỐ LƯỢNG ĐƠN ĐẶT',         # Số lượng sản phẩm của MỖI chi tiết đơn
        'SỐ LƯỢNG SKUS ĐƠN ĐẶT',    # Số chi tiết đơn hàng (số SKUs)
        
        # ❌ BỎ HẾT CÁC CỘT TIỀN:
        # 'TỔNG TIỀN ĐƠN ĐẶT',
        # 'CHIẾT KHẤU ĐƠN ĐẶT',
        # 'TỔNG TRẢ THƯỞNG ĐƠN ĐẶT',
        # 'THÀNH TIỀN ĐƠN ĐẶT',
    ]
    
    # Model architecture - ANTI-OVERFITTING
    HIDDEN_DIM = 128        # REDUCED: 256 → 128
    NUM_HEADS = 4
    NUM_LAYERS = 2          # REDUCED: 3 → 2
    DROPOUT = 0.3           # INCREASED: 0.2 → 0.3
    
    # Sequence config
    HISTORY_LEN = 15        # REDUCED: 20 → 15
    FORECAST_LEN = 1
    
    # Training - STRONGER REGULARIZATION
    BATCH_SIZE = 128        # INCREASED: 64 → 128
    EPOCHS = 100
    LR = 2e-4               # LOWER: 3e-4 → 2e-4
    WEIGHT_DECAY = 1e-3     # INCREASED: 1e-4 → 1e-3
    GRAD_CLIP = 0.5         # REDUCED: 1.0 → 0.5
    EARLY_STOP_PATIENCE = 15
    
    # Data split - BETTER BALANCE
    TRAIN_SPLIT = 0.60      # 60%
    VAL_SPLIT = 0.20        # 20% (tăng từ 15%)
    TEST_SPLIT = 0.20       # 20%
    
    # System
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 2026
    NUM_WORKERS = 0
    
    @classmethod
    def init_dirs(cls):
        """Tạo thư mục"""
        for d in [cls.DATA_DIR, cls.PROCESSED, cls.MODELS, cls.LOGS]:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def summary(cls):
        """In thông tin"""
        print("\n" + "="*80)
        print("CUSTOMER PURCHASE ORDER PREDICTION")
        print("Dự đoán đơn đặt hàng - ĐƠN ĐẶT (NO MONEY)")
        print("="*80)
        print(f"Architecture: TFT (Anti-Overfitting)")
        print(f"Device: {cls.DEVICE}")
        print(f"History length: {cls.HISTORY_LEN} orders")
        print(f"Hidden dim: {cls.HIDDEN_DIM}")
        print(f"Dropout: {cls.DROPOUT}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Weight decay: {cls.WEIGHT_DECAY}")
        print(f"Data split: {cls.TRAIN_SPLIT:.0%}/{cls.VAL_SPLIT:.0%}/{cls.TEST_SPLIT:.0%}")
        print(f"Columns: {len(cls.REQUIRED_COLS)}")
        print("\n📝 Logic:")
        print("  - SỐ LƯỢNG SKUS ĐƠN ĐẶT: Số chi tiết đơn (số dòng)")
        print("  - SỐ LƯỢNG ĐƠN ĐẶT: Số lượng sản phẩm mỗi chi tiết")
        print("="*80 + "\n")


# Set seed
torch.manual_seed(SystemConfig.SEED)

if __name__ == "__main__":
    SystemConfig.init_dirs()
    SystemConfig.summary()