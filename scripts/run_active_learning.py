"""
This script is used to run the active learning pipeline. It loads the model, preprocesses the data, and creates raw 
predictions that can subsequently be used to label the weakest samples.
"""

import os
from desertification_mali.inference.predict import load_model, preprocess_data, predict
import torch
import pandas as pd
import re

def main():
    data_path = 'data/patches/active_learning'
    results_dir = os.makedirs('results', exist_ok=True) or 'results'
    model_path = 'models/final_model_bs8_lr0.0039623270267808855_epochs20_l20.08461373327517657.pth'

    dataset = preprocess_data(data_path)
    output_fp = os.path.join(results_dir, 'activity_learning_predictions.csv')
    print(output_fp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    predictions = predict(model, dataset, device, True)
 
    for prediction in predictions:
        prediction['Prediction'] = prediction['Prediction'].squeeze()

    predictions.sort(key=lambda x: x['Prediction'])
    df = pd.DataFrame(predictions)
    df.to_csv(output_fp, index=False)
    print(f"Predictions saved to {output_fp}")

if __name__ == "__main__":
    main()