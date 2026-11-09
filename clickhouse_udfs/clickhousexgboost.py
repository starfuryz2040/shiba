#!/shiba_env/bin/python3

import sys

import os

#os.environ["OMP_NUM_THREADS"] = "4"
#os.environ["MKL_NUM_THREADS"] = "4"
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
#os.environ["NUMEXPR_NUM_THREADS"] = "4"



import numpy as np
import sys
import json
import xgboost as xgb
import pandas as pd

def predict_xgboost(gender, ethnicity, driver_class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs):

    xgboostmodel = xgb.XGBClassifier(enable_categorical=True)
    xgboostmodel.load_model('/models/xgboost_model.model')

    #xgb.config_context(nthread=1)

    data = [
        {
            'gender': gender,
            'ethnicity': ethnicity,
            'class': driver_class,
            'eye_color': eye_color,
            'hair_color': hair_color,
            'is_donor': is_donor,
            'is_veteran': is_veteran,
            'height_inches': height_inches,
            'weight_lbs': weight_lbs
        }
    ]
    df = pd.DataFrame(data)

    df['gender'] = df['gender'].astype('category')
    df['ethnicity'] = df['ethnicity'].astype('category')
    df['class'] = df['class'].astype('category')
    df['eye_color'] = df['eye_color'].astype('category')
    df['hair_color'] = df['hair_color'].astype('category')
    df['is_donor'] = df['is_donor'].astype('category')
    df['is_veteran'] = df['is_veteran'].astype('category')
    df['height_inches'] = df['height_inches'].astype('int64')
    df['weight_lbs'] = df['weight_lbs'].astype('int64')

    #feature_np = np.array(feature_list, dtype=object)
    #xgb.config_context(nthread=4)
    y_pred = xgboostmodel.predict(df)
    #print(y_pred[0])

    return float(y_pred[0])

if __name__ == '__main__':
    for line in sys.stdin:
        gender, ethnicity, driver_class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs = line.split('\t')
        print(predict_xgboost(gender, ethnicity, driver_class, eye_color, hair_color, is_donor, is_veteran, height_inches, weight_lbs), end='\n')
        #print("Values from 2: " + gender + ' ' + ethnicity + ' ' + driver_class + ' ' + eye_color + ' ' + hair_color + ' ' + is_donor + ' ' + is_veteran + ' ' + height_inches, end='')
        sys.stdout.flush()



