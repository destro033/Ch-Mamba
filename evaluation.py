import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_preprocessing_patchtst import TimeSeriesDataset


def load_test_dataframe(csv_path, sep=";"):
    return pd.read_csv(csv_path, sep=sep)


def prepare_flight_loader(
    df,
    flight_id,
    feature_cols,
    scaler,
    input_length=96,
    forecast_length=96,
    batch_size=32
):
    data_raw = df.loc[df["uid"] == flight_id, feature_cols].values
    data_scaled = scaler.transform(data_raw)

    dataset = TimeSeriesDataset(data_scaled, input_length, forecast_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def predict_for_flight(model, loader, device):
    preds_all, targets_all = [], []

    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X, None, None, None).cpu()
            preds_all.append(preds)
            targets_all.append(y.cpu())

    y_pred = torch.cat(preds_all, dim=0)   # (N, F, V)
    y_true = torch.cat(targets_all, dim=0) # (N, F, V)

    return y_pred, y_true


def inverse_scale_predictions(y_pred, y_true, scaler):
    N, F, V = y_pred.shape

    y_pred_2d = y_pred.reshape(-1, V)
    y_true_2d = y_true.reshape(-1, V)

    y_pred_real = scaler.inverse_transform(y_pred_2d)
    y_true_real = scaler.inverse_transform(y_true_2d)

    y_pred_real = y_pred_real.reshape(N, F, V)
    y_true_real = y_true_real.reshape(N, F, V)

    return y_pred_real, y_true_real


def compute_mae_for_flight(
    model,
    df,
    flight_id,
    feature_cols,
    scaler,
    device,
    input_length=96,
    forecast_length=96,
    batch_size=32
):
    loader = prepare_flight_loader(
        df, flight_id, feature_cols, scaler, input_length, forecast_length, batch_size
    )
    y_pred, y_true = predict_for_flight(model, loader, device)
    y_pred_real, y_true_real = inverse_scale_predictions(y_pred, y_true, scaler)

    abs_error = np.abs(y_pred_real - y_true_real)
    mae_xyz = abs_error.mean(axis=(0, 1))   # keep variable axis
    latitude_rad = np.deg2rad(y_true_real[:, :, 1].mean())

    mae_meters = np.array([
        mae_xyz[0] * 111320 * np.cos(latitude_rad),
        mae_xyz[1] * 111320,
        mae_xyz[2]
    ])

    return mae_meters


def compute_ade_for_flight(
    model,
    df,
    flight_id,
    feature_cols,
    scaler,
    device,
    input_length=96,
    forecast_length=96,
    batch_size=32
):
    loader = prepare_flight_loader(
        df, flight_id, feature_cols, scaler, input_length, forecast_length, batch_size
    )
    y_pred, y_true = predict_for_flight(model, loader, device)
    y_pred_real, y_true_real = inverse_scale_predictions(y_pred, y_true, scaler)

    lat_deg = np.mean(y_true_real[:, :, 1])
    lat_rad = np.deg2rad(lat_deg)

    disp = y_pred_real - y_true_real
    disp[:, :, 0] *= 111320 * np.cos(lat_rad)
    disp[:, :, 1] *= 111320

    l2_dist_first = np.linalg.norm(disp, axis=2)   # (N, F)
    ade_first = l2_dist_first.mean()
    
    return ade_first


def compute_euclidean_error_per_timestep(
    model,
    df,
    flight_id,
    feature_cols,
    scaler,
    device,
    input_length=96,
    forecast_length=96,
    batch_size=32
):
    loader = prepare_flight_loader(
        df, flight_id, feature_cols, scaler, input_length, forecast_length, batch_size
    )
    y_pred, y_true = predict_for_flight(model, loader, device)
    y_pred_real, y_true_real = inverse_scale_predictions(y_pred, y_true, scaler)

    lat_deg = np.mean(y_true_real[:, :, 1])
    lat_rad = np.deg2rad(lat_deg)

    disp = y_pred_real - y_true_real
    disp[:, :, 0] *= 111320 * np.cos(lat_rad)
    disp[:, :, 1] *= 111320

    l2_per_timestep = np.linalg.norm(disp, axis=2)   # (N, F)
    errors_per_timestep = l2_per_timestep.mean(axis=0)

    return errors_per_timestep


def get_trajectory_for_plot(
    model,
    df,
    flight_id,
    feature_cols,
    scaler,
    device,
    input_length=96,
    forecast_length=96,
    batch_size=32
):
    loader = prepare_flight_loader(
        df, flight_id, feature_cols, scaler, input_length, forecast_length, batch_size
    )
    y_pred, y_true = predict_for_flight(model, loader, device)
    y_pred_real, y_true_real = inverse_scale_predictions(y_pred, y_true, scaler)

    return y_pred_real, y_true_real


def plot_mae_bars(mae_17, mae_18):
    labels = ['X', 'Y', 'Z']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width / 2, mae_17, width, label='Complicated Flight', color='blue')
    bars2 = ax.bar(x + width / 2, mae_18, width, label='Simple Flight', color='red')

    ax.set_ylabel('MAE (meters)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    max_val = max(np.max(mae_17), np.max(mae_18))
    ax.set_ylim(0, max_val * 1.15)

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9
            )

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    return fig, ax


def plot_ade_bars(ade_17, ade_18):
    flights = ['Complicated Flight', 'Simple Flight']
    ade_values = [ade_17, ade_18]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(flights, ade_values, color=['blue', 'red'])

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_ylabel('Euclidean distance (meters)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_error_vs_forecast(error_17, error_18):
    forecast_steps = np.arange(1, len(error_17) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(forecast_steps, error_17, label='Complicated Flight', color='blue')
    ax.plot(forecast_steps, error_18, label='Simple Flight', color='red')
    ax.set_xlabel('Forecast Step')
    ax.set_ylabel('Euclidean Error (meters)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_3d_trajectory(y_pred_real, y_true_real, forecast_step=0):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.ticker import MaxNLocator

    x_idx, y_idx, z_idx = 0, 1, 2

    x_true = y_true_real[:, forecast_step, x_idx]
    y_true = y_true_real[:, forecast_step, y_idx]
    z_true = y_true_real[:, forecast_step, z_idx]

    x_pred = y_pred_real[:, forecast_step, x_idx]
    y_pred = y_pred_real[:, forecast_step, y_idx]
    z_pred = y_pred_real[:, forecast_step, z_idx]

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_true, y_true, z_true, label='Ground Truth', linewidth=2)
    ax.plot(x_pred, y_pred, z_pred, label='Prediction')

    ax.ticklabel_format(style='plain', useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))

    ax.set_xlabel('X (deg)', labelpad=10)
    ax.set_ylabel('Y (deg)', labelpad=10)
    ax.set_zlabel('Z (m)', labelpad=0.05)
    ax.legend()

    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    return fig, ax
