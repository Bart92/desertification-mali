from desertification_mali.inference.predict import load_model, predict, preprocess_data
import torch


def main():
    model_path = 'models/final_model_bs8_lr0.008841921163357549_epochs10_l20.09466658850904519.pth'
    data_path = 'data/patches/manual_labeling'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    dataset = preprocess_data(data_path)
    predictions = predict(model, dataset, device)

    for i, prediction in enumerate(predictions):
        print(f"Sample {i+1}: Prediction: {prediction}")

if __name__ == "__main__":
    main()