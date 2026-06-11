import os
import pandas as pd
from src.config import Config

try:
    # 1. Validate paths
    Config.validate_paths()
    
    # 2. Check API Key
    if Config.GOOGLE_API_KEY:
        masked_key = Config.GOOGLE_API_KEY[:5] + "..." + Config.GOOGLE_API_KEY[-4:] if len(Config.GOOGLE_API_KEY) > 10 else "Too short"
        print(f"API Key Found: {masked_key}")
    else:
        print("API Key: Not Found")
        
    # 3. Check if Dataset is readable
    df = pd.read_csv(Config.DATASET_PATH)
    print(f"Dataset successfully loaded. Total records found: {len(df)}")
    print("Verification complete! System is ready.")
    
except Exception as e:
    print("Verification failed with error:")
    print(str(e))