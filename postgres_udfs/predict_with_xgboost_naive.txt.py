create function predict_with_xgboost_naive(gender text, ethnicity text, driver_class text, eye_color text, hair_color text, is_donor text, is_veteran text, height_inches integer, weight_lbs integer) returns text
    language plpython3u
as
$$
import pandas as pd
import numpy as np
import xgboost as xgb


try:
    xgboostmodel_optimized = xgb.XGBClassifier(enable_categorical=True)
    xgboostmodel_optimized.load_model('/models/xgboost_model.model')

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

    y_pred = xgboostmodel_optimized.predict(df)

    return float(y_pred)
except Exception as e:
    return f"Error during inference: {e}"
$$;

alter function predict_with_xgboost_naive(text, text, text, text, text, text, text, integer, integer) owner to postgres;

