import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_length, forecast_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_length = input_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data) - self.input_length - self.forecast_length + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_length]
        y = self.data[idx + self.input_length: idx + self.input_length + self.forecast_length]

        x = x.T
        y = y.T

        return x, y


def create_dataloaders_from_csv(csv_path, input_length=96, forecast_length=96, batch_size=32):
    df = pd.read_csv(csv_path, sep=";")

    features = df[['position_x', 'position_y', 'position_z']]
    feature_cols = features.columns

    train_raw = df.loc[df["uid"].between(1, 12), feature_cols]
    val_raw   = df.loc[df["uid"].between(13, 16), feature_cols]

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_raw)
    val_data = scaler.transform(val_raw)

    train_dataset = TimeSeriesDataset(train_data, input_length, forecast_length)
    val_dataset = TimeSeriesDataset(val_data, input_length, forecast_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler
