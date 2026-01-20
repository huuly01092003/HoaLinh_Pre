print("Starting...")
import sys
sys.stdout.flush()

print("Step 1: Import pandas")
sys.stdout.flush()
import pandas as pd

print("Step 2: Load first 1000 rows")
sys.stdout.flush()
df = pd.read_csv('../data/raw/merged_2025.csv', encoding='utf-8-sig', nrows=1000)

print(f"SUCCESS! Loaded {len(df)} rows")
print(f"Columns: {len(df.columns)}")