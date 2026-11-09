#!/shiba_env/bin/python3

import sys
import os

# Limit underlying multi-threading layout to prevent core fighting
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import xgboost as xgb
import pandas as pd

def main():
    # 1. Initialize and load the XGBoost model ONCE at process startup
    xgboostmodel = xgb.XGBClassifier(enable_categorical=True)
    xgboostmodel.load_model('/models/xgboost_model.model')
    
    # Configure global XGBoost threading
    xgb.config_context(nthread=1)

    while True:
        line = sys.stdin.readline()
        if not line:
            break  # Server closed stream or terminated
        
        # 2. Extract the chunk size header integer sent by ClickHouse
        num_rows = int(line.strip())
        
        # 3. Read the exact number of allocated rows for this batch chunk
        batch_data = []
        for _ in range(num_rows):
            row_line = sys.stdin.readline().rstrip('\n')
            columns = row_line.split('\t')
            
            # Map the split columns into a flat dictionary structure
            batch_data.append({
                'gender': columns[0],
                'ethnicity': columns[1],
                'class': columns[2],
                'eye_color': columns[3],
                'hair_color': columns[4],
                'is_donor': columns[5],
                'is_veteran': columns[6],
                'height_inches': columns[7],
                'weight_lbs': columns[8]
            })
            
        # 4. Safely convert the entire batch into a structured Pandas DataFrame at once
        df = pd.DataFrame(batch_data)

        # Enforce schemas over the columns for XGBoost compatibility
        df['gender'] = df['gender'].astype('category')
        df['ethnicity'] = df['ethnicity'].astype('category')
        df['class'] = df['class'].astype('category')
        df['eye_color'] = df['eye_color'].astype('category')
        df['hair_color'] = df['hair_color'].astype('category')
        df['is_donor'] = df['is_donor'].astype('category')
        df['is_veteran'] = df['is_veteran'].astype('category')
        df['height_inches'] = df['height_inches'].astype('int64')
        df['weight_lbs'] = df['weight_lbs'].astype('int64')

        # 5. Execute true vectorized inference over the whole batch
        y_preds = xgboostmodel.predict(df)
        
        # 6. Stream predictions back sequentially in the exact order received
        for pred in y_preds:
            sys.stdout.write(f"{float(pred)}\n")
            
        sys.stdout.flush()

if __name__ == '__main__':
    main()

