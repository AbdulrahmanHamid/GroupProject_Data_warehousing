# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:30:00 2025

@author: group2


The API receives JSON data, One-Hot encodes it, aligns columns
to match training data, scales the data, and returns a prediction.
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
import os

app = Flask(__name__)

# Load components
path = "C:\\Users\\abdul\\OneDrive\\Desktop\\Centennial S\\Semester 5\\data wherehousing\\GroupProject_Data_warehousing\\"
try:
    print("Loading model components...")
    model = joblib.load(os.path.join(path, 'model_group2.pkl'))
    model_columns = joblib.load(os.path.join(path, 'model_columns_group2.pkl'))
    scaler = joblib.load(os.path.join(path, 'model_scaler_group2.pkl'))
    le = joblib.load(os.path.join(path, 'model_encoder_group2.pkl'))
    print("Model loaded successfully.")
except:
    model = None
    print("Model failed to load.")

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            
            # 1. Get JSON data
            json_ = request.json
            print("\nReceived Input:", json_)
            
            # 2. Convert to DataFrame
            query_df = pd.DataFrame([json_])
            
            # 3. to convert "Feature Time of Day" 
            if 'OCC_HOUR' in query_df.columns:
                query_df["TIME_OF_DAY"] = pd.cut(
                    query_df["OCC_HOUR"],
                    bins=[0, 6, 12, 18, 24],
                    labels=["Night", "Morning", "Afternoon", "Evening"],
                    include_lowest=True
                )
                print(f"Converted Hour {query_df['OCC_HOUR'][0]} -> {query_df['TIME_OF_DAY'][0]}")

            # 4. One-Hot Encode
            query_df = pd.get_dummies(query_df)
            
            # 5. Align columns
            query_df = query_df.reindex(columns=model_columns, fill_value=0)
            
            # 6. Scale data 
            scaled_data = scaler.transform(query_df)
            query_scaled = pd.DataFrame(scaled_data, columns=model_columns)
            
            # 7. Predict
            prediction_idx = model.predict(query_scaled)
            prediction_label = le.inverse_transform(prediction_idx)
            
            result = {
                'prediction_code': int(prediction_idx[0]), 
                'prediction_label': prediction_label[0]
            }
            print("Result:", result)
            return jsonify(result)

        except:
            return jsonify({'trace': traceback.format_exc()})
        
    else:
        
        return ('No model found. Check file paths.')


if __name__ == '__main__':
    
    
    app.run(port=12345, debug=False)