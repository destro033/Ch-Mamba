import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_preprocessing import TimeSeriesDataset


def load_test_dataframe(csv_path, sep=";"):
    return pd.read_csv(csv_path, sep=sep)



def evaluate_flights_full(
    model,
    df,
    flight_ids,
    feature_cols,
    scaler,
    device,
    input_length=96,
    forecast_length=96,
    batch_size=32,
    save_path="evaluation_results.npz"
):
    
    all_preds = []
    all_trues = []
    flight_sample_counts = []

    model.eval()

    with torch.no_grad():
        for flight_id in flight_ids:

            data_raw = df.loc[df["uid"] == flight_id, feature_cols].values
            data_scaled = scaler.transform(data_raw)

            dataset = TimeSeriesDataset(
                data_scaled,
                input_length,
                forecast_length
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )

            preds_all = []
            targets_all = []

            for X, y in loader:
                X = X.to(device)

                preds = model(X, None, None, None).cpu()

                preds_all.append(preds)
                targets_all.append(y.cpu())

            y_pred_flight = torch.cat(preds_all, dim=0)
            y_true_flight = torch.cat(targets_all, dim=0)

            all_preds.append(y_pred_flight)
            all_trues.append(y_true_flight)

            # save how many samples this flight produced
            flight_sample_counts.append(y_pred_flight.shape[0])

    # combine all flights in order
    # [flight 1 samples][flight 2 samples]
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_trues, dim=0)

    # inverse scaling
    N, F, V = y_pred.shape #(num of samples, sequence length, num of variables)

    y_pred_2d = y_pred.reshape(-1, V)
    y_true_2d = y_true.reshape(-1, V)

    y_pred_real = scaler.inverse_transform(y_pred_2d)
    y_true_real = scaler.inverse_transform(y_true_2d)

    y_pred_real = y_pred_real.reshape(N, F, V)
    y_true_real = y_true_real.reshape(N, F, V)

    # displacement in original units
    disp = y_pred_real - y_true_real

    # convert degrees to meters first
    lat_rad = np.deg2rad(y_true_real[:, :, 1])

    disp_meters = disp.copy()
    disp_meters[:, :, 0] *= 111320 * np.cos(lat_rad)
    disp_meters[:, :, 1] *= 111320
    # Z stays unchanged as it is in meters

    # MAE for X, Y, Z in meters
    mae_xyz = np.abs(disp_meters).mean(axis=(0, 1))

    # Euclidean error for every sample and forecast step
    euclidean_errors = np.linalg.norm(disp_meters, axis=2)

    # Euclidean error for first forecast step
    ade_first_step = euclidean_errors[:, 0].mean()

    # ADE over all forecast steps
    ade_96 = euclidean_errors.mean()

    # Error curve over forecast horizon
    error_per_forecast_step = euclidean_errors.mean(axis=0)

    # save everything needed for later plotting
    np.savez(
        save_path,
        mae_xyz=mae_xyz,
        ade_first_step=ade_first_step,
        ade_96=ade_96,
        error_per_forecast_step=error_per_forecast_step,
        euclidean_errors=euclidean_errors,
        preds=y_pred_real,
        gt=y_true_real,
        disp_meters=disp_meters,
        flight_ids=np.array(flight_ids),
        flight_sample_counts=np.array(flight_sample_counts)
    )





