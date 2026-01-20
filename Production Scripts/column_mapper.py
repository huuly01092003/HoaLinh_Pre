"""
Column Name Mapper
Maps actual CSV column names to code-expected names
"""

# Actual column names in CSV → Expected names in code
COLUMN_MAPPING = {
    # Customer info
    'MÃ KHÁCH HÀNG': 'MÃ KHÁCH HÀNG',
    'TÊN KHÁCH HÀNG': 'TÊN KHÁCH HÀNG',
    'LÀ KHÁCH HÀNG VÃNG LAI': 'LÀ KHÁCH HÀNG VÃNG LAI',
    
    # Location
    'TỈNH/TP CỦA KHÁCH HÀNG': 'TỈNH/TP CỦA KHÁCH HÀNG',
    'QUẬN/HUYỆN CỦA KHÁCH HÀNG': 'QUẬN/HUYỆN CỦA KHÁCH HÀNG',
    'ĐỊA CHỈ CỦA KHÁCH HÀNG': 'ĐỊA CHỈ CỦA KHÁCH HÀNG',
    
    # Employee & Route
    'MÃ NHÂN VIÊN': 'MÃ NHÂN VIÊN',
    'TUYẾN BÁN HÀNG': 'TUYẾN BÁN HÀNG',
    
    # Order info - ĐƠN BÁN (not ĐƠN ĐẶT)
    'MÃ ĐƠN BÁN HÀNG': 'MÃ ĐƠN BÁN HÀNG',
    'NGÀY TẠO ĐƠN BÁN': 'NGÀY TẠO ĐƠN BÁN',
    'THỜI GIAN TẠO ĐƠN BÁN': 'THỜI GIAN TẠO ĐƠN BÁN',
    
    # Product info - ĐƠN BÁN
    'MÃ SẢN PHẨM ĐƠN BÁN': 'MÃ SẢN PHẨM ĐƠN BÁN',
    'LOẠI SẢN PHẨM': 'LOẠI SẢN PHẨM',
    
    # Numbers - ĐƠN BÁN
    'SỐ LƯỢNG ĐƠN BÁN': 'SỐ LƯỢNG ĐƠN BÁN',
    'ĐƠN GIÁ ĐƠN BÁN': 'ĐƠN GIÁ ĐƠN BÁN',
    'TỔNG TIỀN ĐƠN BÁN': 'TỔNG TIỀN ĐƠN BÁN',
    'CHIẾT KHẤU ĐƠN BÁN': 'CHIẾT KHẤU ĐƠN BÁN',
    'THÀNH TIỀN ĐƠN BÁN': 'THÀNH TIỀN ĐƠN BÁN',
    'CK SP ĐƠN BÁN': 'CK SP ĐƠN BÁN',
    'CK ĐƠN HÀNG ĐƠN BÁN': 'CK ĐƠN HÀNG ĐƠN BÁN',
    'TỔNG CHIẾT KHẤU ĐƠN BÁN': 'TỔNG CHIẾT KHẤU ĐƠN BÁN',
    
    # Source
    'NGUỒN ĐƠN': 'NGUỒN ĐƠN',
}


def map_columns(df):
    """
    Map CSV column names to expected names
    Only keeps columns that are in the mapping
    """
    # Filter to only keep columns we need
    available_cols = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
    df_filtered = df[available_cols].copy()
    
    # Rename columns (in this case, no change needed as they match)
    # But this allows for flexibility if names differ
    
    return df_filtered


def print_column_comparison(df):
    """Print comparison of CSV columns vs expected columns"""
    print("\n" + "="*80)
    print("COLUMN COMPARISON")
    print("="*80)
    
    csv_cols = set(df.columns)
    expected_cols = set(COLUMN_MAPPING.keys())
    
    missing = expected_cols - csv_cols
    extra = csv_cols - expected_cols
    matched = csv_cols & expected_cols
    
    print(f"\n✓ Matched columns: {len(matched)}")
    for col in sorted(matched):
        print(f"  - {col}")
    
    if missing:
        print(f"\n⚠️  Missing columns (expected but not found): {len(missing)}")
        for col in sorted(missing):
            print(f"  - {col}")
    
    if extra:
        print(f"\n📋 Extra columns (in CSV but not used): {len(extra)}")
        # Only show first 10 to avoid clutter
        for col in sorted(list(extra))[:10]:
            print(f"  - {col}")
        if len(extra) > 10:
            print(f"  ... and {len(extra)-10} more")
    
    print("="*80 + "\n")