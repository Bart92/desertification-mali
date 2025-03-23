"""
This script is used to run inference on the model.
"""

from desertification_mali.inference.predict import load_model, predict, preprocess_data
import torch
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference for the Siamese Network.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    args = parser.parse_args()

    data_path = 'data/patches/active_learning'
    results_dir = os.makedirs('results', exist_ok=True) or 'results'
    results_file = os.path.join(results_dir, 'predictions.csv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)
    dataset = preprocess_data(data_path)
    predictions = predict(model, dataset, device)

    df = pd.DataFrame(predictions)
    df.to_csv(results_file, index=False)
    
    print(f"Predictions saved to {results_file}")

if __name__ == "__main__":
    main()