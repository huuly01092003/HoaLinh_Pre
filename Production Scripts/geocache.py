"""
Geographic Caching System for Vietnam's 34 Provinces
Based on Nghị quyết 202/2025/QH15 (effective 12/6/2025)
FIXED: Removed tqdm import that was causing hang
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict


class GeoCache:
    """Optimized geocoding cache for Vietnam's 34 provinces"""
    
    def __init__(self, cache_file: Optional[Path] = None):
        self.address_cache = {}
        self.province_cache = {}
        self.cache_file = cache_file
        
        # 34 provinces after 2025 merger
        self.province_coords = {
            # 11 unchanged provinces
            'Thành phố Hà Nội': (21.0285, 105.8542),
            'Thành phố Huế': (16.4637, 107.5905),
            'Tỉnh Lai Châu': (22.3864, 103.4702),
            'Tỉnh Điện Biên': (21.3833, 103.0167),
            'Tỉnh Sơn La': (21.3256, 103.9188),
            'Tỉnh Cao Bằng': (22.6667, 106.2500),
            'Tỉnh Lạng Sơn': (21.8536, 106.7619),
            'Tỉnh Quảng Ninh': (21.0064, 107.2925),
            'Tỉnh Thanh Hóa': (19.8067, 105.7851),
            'Tỉnh Nghệ An': (18.6792, 105.6819),
            'Tỉnh Hà Tĩnh': (18.3559, 105.8877),
            
            # 23 merged provinces
            'Tỉnh Tuyên Quang': (21.8236, 105.2280),
            'Tỉnh Lào Cai': (22.4856, 103.9707),
            'Tỉnh Thái Nguyên': (21.5671, 105.8252),
            'Tỉnh Phú Thọ': (21.2680, 105.2045),
            'Tỉnh Bắc Ninh': (21.1861, 106.0763),
            'Tỉnh Hưng Yên': (20.6464, 106.0511),
            'Thành phố Hải Phòng': (20.8449, 106.6881),
            'Tỉnh Ninh Bình': (20.2506, 105.9745),
            'Tỉnh Quảng Trị': (16.7404, 107.1854),
            'Thành phố Đà Nẵng': (16.0544, 108.2022),
            'Tỉnh Quảng Ngãi': (15.1214, 108.8044),
            'Tỉnh Gia Lai': (13.9830, 108.0005),
            'Tỉnh Khánh Hòa': (12.2388, 109.1967),
            'Tỉnh Lâm Đồng': (11.9404, 108.4583),
            'Tỉnh Đắk Lắk': (12.7100, 108.2378),
            'Thành phố Hồ Chí Minh': (10.8231, 106.6297),
            'Tỉnh Đồng Nai': (10.9517, 106.8442),
            'Tỉnh Tây Ninh': (11.3351, 106.0983),
            'Thành phố Cần Thơ': (10.0452, 105.7469),
            'Tỉnh Vĩnh Long': (10.2395, 105.9572),
            'Tỉnh Đồng Tháp': (10.4938, 105.6881),
            'Tỉnh Cà Mau': (9.1526, 105.1960),
            'Tỉnh An Giang': (10.5215, 105.1258),
        }
        
        # Mapping from old 63 provinces to new 34
        self.old_to_new_mapping = self._build_mapping()
        
        # Load cache if exists
        if cache_file and cache_file.exists():
            self.load_cache()
        
        pass
    
    def _build_mapping(self) -> Dict[str, str]:
        """Build complete old-to-new province mapping"""
        mapping = {}
        
        # Unchanged provinces (11)
        unchanged = [
            'Thành phố Hà Nội', 'Thành phố Huế', 'Tỉnh Lai Châu',
            'Tỉnh Điện Biên', 'Tỉnh Sơn La', 'Tỉnh Cao Bằng',
            'Tỉnh Lạng Sơn', 'Tỉnh Quảng Ninh', 'Tỉnh Thanh Hóa',
            'Tỉnh Nghệ An', 'Tỉnh Hà Tĩnh'
        ]
        for prov in unchanged:
            mapping[prov] = prov
        
        # Merged provinces
        mergers = {
            'Tỉnh Tuyên Quang': ['Tỉnh Tuyên Quang', 'Tỉnh Yên Bái'],
            'Tỉnh Lào Cai': ['Tỉnh Lào Cai', 'Tỉnh Hà Giang'],
            'Tỉnh Thái Nguyên': ['Tỉnh Thái Nguyên', 'Tỉnh Bắc Kạn'],
            'Tỉnh Phú Thọ': ['Tỉnh Phú Thọ', 'Tỉnh Bắc Giang', 'Tỉnh Hòa Bình'],
            'Tỉnh Bắc Ninh': ['Tỉnh Bắc Ninh', 'Tỉnh Vĩnh Phúc'],
            'Tỉnh Hưng Yên': ['Tỉnh Hưng Yên', 'Tỉnh Hải Dương', 'Tỉnh Hà Nam'],
            'Thành phố Hải Phòng': ['Thành phố Hải Phòng', 'Tỉnh Thái Bình', 'Tỉnh Nam Định'],
            'Tỉnh Ninh Bình': ['Tỉnh Ninh Bình'],
            'Tỉnh Quảng Trị': ['Tỉnh Quảng Trị', 'Tỉnh Quảng Bình'],
            'Thành phố Đà Nẵng': ['Thành phố Đà Nẵng', 'Tỉnh Quảng Nam'],
            'Tỉnh Quảng Ngãi': ['Tỉnh Quảng Ngãi', 'Tỉnh Bình Định', 'Tỉnh Phú Yên'],
            'Tỉnh Gia Lai': ['Tỉnh Gia Lai', 'Tỉnh Kon Tum'],
            'Tỉnh Khánh Hòa': ['Tỉnh Khánh Hòa', 'Tỉnh Ninh Thuận', 'Tỉnh Bình Thuận'],
            'Tỉnh Lâm Đồng': ['Tỉnh Lâm Đồng'],
            'Tỉnh Đắk Lắk': ['Tỉnh Đắk Lắk', 'Tỉnh Đắk Nông'],
            'Thành phố Hồ Chí Minh': ['Thành phố Hồ Chí Minh'],
            'Tỉnh Đồng Nai': ['Tỉnh Đồng Nai', 'Tỉnh Bà Rịa - Vũng Tàu'],
            'Tỉnh Tây Ninh': ['Tỉnh Tây Ninh', 'Tỉnh Bình Dương', 'Tỉnh Bình Phước', 
                              'Tỉnh Long An', 'Tỉnh Tiền Giang'],
            'Thành phố Cần Thơ': ['Thành phố Cần Thơ', 'Tỉnh Hậu Giang', 
                                   'Tỉnh Sóc Trăng', 'Tỉnh Bạc Liêu'],
            'Tỉnh Vĩnh Long': ['Tỉnh Vĩnh Long', 'Tỉnh Bến Tre', 'Tỉnh Trà Vinh'],
            'Tỉnh Đồng Tháp': ['Tỉnh Đồng Tháp'],
            'Tỉnh Cà Mau': ['Tỉnh Cà Mau'],
            'Tỉnh An Giang': ['Tỉnh An Giang', 'Tỉnh Kiên Giang'],
        }
        
        for new_prov, old_provs in mergers.items():
            for old_prov in old_provs:
                mapping[old_prov] = new_prov
        
        return mapping
    
    def normalize_province_name(self, province: str) -> Optional[str]:
        """Normalize province name to new 34-province structure"""
        if pd.isna(province):
            return None
        
        province_str = str(province).strip()
        clean_name = province_str.replace('Thành phố ', '').replace('Tỉnh ', '').strip()
        
        # Direct match
        for key in self.province_coords.keys():
            key_clean = key.replace('Thành phố ', '').replace('Tỉnh ', '').strip()
            if clean_name.lower() == key_clean.lower():
                return key
        
        # Check mapping
        if province_str in self.old_to_new_mapping:
            return self.old_to_new_mapping[province_str]
        
        # Try with prefixes
        for prefix in ['Thành phố ', 'Tỉnh ', '']:
            test_name = f"{prefix}{clean_name}"
            if test_name in self.old_to_new_mapping:
                return self.old_to_new_mapping[test_name]
        
        return None
    
    def geocode_address(self, address: str, district: str, province: str) -> Tuple[float, float]:
        """Geocode address with caching"""
        cache_key = f"{address}|{district}|{province}"
        
        if cache_key in self.address_cache:
            return self.address_cache[cache_key]
        
        normalized_province = self.normalize_province_name(province)
        
        if normalized_province and normalized_province in self.province_coords:
            base_lat, base_lon = self.province_coords[normalized_province]
        else:
            base_lat, base_lon = 21.0285, 105.8542  # Hanoi fallback
        
        # Add offset based on address hash
        if pd.notna(address) and str(address).strip():
            addr_hash = int(hashlib.md5(str(address).encode()).hexdigest()[:8], 16)
            lat_offset = ((addr_hash % 1000) - 500) / 5000
            lon_offset = (((addr_hash // 1000) % 1000) - 500) / 5000
            coords = (base_lat + lat_offset, base_lon + lon_offset)
        else:
            coords = (base_lat, base_lon)
        
        self.address_cache[cache_key] = coords
        return coords
    
    def get_province_center(self, province: str) -> Tuple[float, float]:
        """Get province center coordinates"""
        normalized_province = self.normalize_province_name(province)
        if normalized_province and normalized_province in self.province_coords:
            return self.province_coords[normalized_province]
        return (21.0285, 105.8542)
    
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
        """Calculate haversine distance in km"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def batch_geocode(self, addresses_df: pd.DataFrame) -> list:
        """
        Batch geocode addresses
        FIXED: Removed tqdm, added simple progress logging
        """
        if addresses_df is None or len(addresses_df) == 0:
            return []
        
        results = []
        total = len(addresses_df)
        
        # Get column names (handle both Vietnamese and English)
        addr_col = None
        dist_col = None
        prov_col = None
        
        for col in addresses_df.columns:
            col_lower = col.lower()
            if 'địa chỉ' in col_lower or 'address' in col_lower:
                addr_col = col
            elif 'quận' in col_lower or 'huyện' in col_lower or 'district' in col_lower:
                dist_col = col
            elif 'tỉnh' in col_lower or 'thành phố' in col_lower or 'province' in col_lower:
                prov_col = col
        
        # Fallback to positional if names not found
        if addr_col is None and len(addresses_df.columns) > 0:
            addr_col = addresses_df.columns[0]
        if dist_col is None and len(addresses_df.columns) > 1:
            dist_col = addresses_df.columns[1]
        if prov_col is None and len(addresses_df.columns) > 2:
            prov_col = addresses_df.columns[2]
        
        # Geocode each address with simple progress
        print(f"Geocoding {total} addresses...")
        for idx, row in addresses_df.iterrows():
            address = row.get(addr_col, '') if addr_col else ''
            district = row.get(dist_col, '') if dist_col else ''
            province = row.get(prov_col, '') if prov_col else ''
            
            coords = self.geocode_address(address, district, province)
            results.append(coords)
            
            # Show progress every 10%
            if (idx + 1) % max(1, total // 10) == 0:
                print(f"  Processed {idx + 1}/{total} ({(idx+1)/total*100:.0f}%)")
        
        print(f"✓ Geocoding complete: {total} addresses")
        return results
    
    def save_cache(self):
        """Save cache to disk"""
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'address_cache': self.address_cache,
                    'province_cache': self.province_cache
                }, f)
            print(f"✓ Cache saved: {len(self.address_cache):,} addresses")
    
    def load_cache(self):
        """Load cache from disk"""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.address_cache = data.get('address_cache', {})
                self.province_cache = data.get('province_cache', {})
            print(f"✓ Cache loaded: {len(self.address_cache):,} addresses")