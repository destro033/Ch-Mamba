import joblib
import torch
import matplotlib.pyplot as plt

from model import CMamba
from arguments import args
from evaluation import (
    load_test_dataframe,
    compute_mae_for_flight,
    compute_ade_for_flight,
    compute_euclidean_error_per_timestep,
    get_trajectory_for_plot,
    plot_mae_bars,
    plot_ade_bars,
    plot_error_vs_forecast,
    plot_3d_trajectory,
)

# =========================
# CHANGE THESE IF NEEDED
# =========================
CSV_PATH = "Drone Onboard Multi-Modal Feature-Based Visual Odometry Dataset.csv"
WEIGHTS_PATH = "weights.pth"
SCALER_PATH = "scaler.pkl"

INPUT_LENGTH = 96
FORECAST_LENGTH = 96
BATCH_SIZE = 32

FLIGHT_1 = 17
FLIGHT_2 = 18

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# Load dataframe
# =========================
df = load_test_dataframe(CSV_PATH, sep=";")
feature_cols = ['position_x', 'position_y', 'position_z']

# =========================
# Load model
# =========================
model = CMamba(args).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# =========================
# Load scaler
# =========================
scaler = joblib.load(SCALER_PATH)

# =========================
# MAE
# =========================
mae_17 = compute_mae_for_flight(
    model, df, FLIGHT_1, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

mae_18 = compute_mae_for_flight(
    model, df, FLIGHT_2, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

plot_mae_bars(mae_17, mae_18)

# =========================
# ADE
# =========================
ade_17 = compute_ade_for_flight(
    model, df, FLIGHT_1, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

ade_18 = compute_ade_for_flight(
    model, df, FLIGHT_2, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

print(f"ADE flight {FLIGHT_1}: {ade_17:.2f} m")
print(f"ADE flight {FLIGHT_2}: {ade_18:.2f} m")

plot_ade_bars(ade_17, ade_18)

# =========================
# Error vs forecast step
# =========================
error_17 = compute_euclidean_error_per_timestep(
    model, df, FLIGHT_1, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

error_18 = compute_euclidean_error_per_timestep(
    model, df, FLIGHT_2, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

plot_error_vs_forecast(error_17, error_18)

# =========================
# 3D trajectory - Flight 17
# =========================
y_pred_17, y_true_17 = get_trajectory_for_plot(
    model, df, FLIGHT_1, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

plot_3d_trajectory(y_pred_17, y_true_17, forecast_step=0)

# =========================
# 3D trajectory - Flight 18
# =========================
y_pred_18, y_true_18 = get_trajectory_for_plot(
    model, df, FLIGHT_2, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

plot_3d_trajectory(y_pred_18, y_true_18, forecast_step=0)

# =========================
# Show all plots at the end
# =========================
plt.show()
