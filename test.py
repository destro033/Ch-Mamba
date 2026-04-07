import joblib
import torch

from model import ModelArgs, CMamba
from evaluation import (
    load_test_dataframe,
    compute_mae_for_flight,
    compute_ade_for_flight,
    compute_euclidean_error_per_timestep,
    compute_cdf_first_forecast,
    get_trajectory_for_plot,
    plot_mae_bars,
    plot_ade_bars,
    plot_error_vs_forecast,
    plot_cdf,
    plot_3d_trajectory,
)

# =========================
# User settings
# =========================
CSV_PATH = "Drone Onboard Multi-Modal Feature-Based Visual Odometry Dataset.csv"
ARGS_PATH = "model_args.pth"
WEIGHTS_PATH = "cmamba_best_model.pth"
SCALER_PATH = "scaler.pkl"

INPUT_LENGTH = 96
FORECAST_LENGTH = 96
BATCH_SIZE = 32

FLIGHT_1 = 17
FLIGHT_2 = 18
TRAJECTORY_FLIGHT = 17

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
# Load args
# =========================
args_dict = torch.load(ARGS_PATH, map_location="cpu")

args_dict.pop("v", None)
args_dict.pop("num_patches", None)
args_dict.pop("d_inner", None)

args = ModelArgs(**args_dict)

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
# 3D trajectory
# =========================
y_pred_real, y_true_real = get_trajectory_for_plot(
    model, df, TRAJECTORY_FLIGHT, feature_cols, scaler, device,
    input_length=INPUT_LENGTH, forecast_length=FORECAST_LENGTH, batch_size=BATCH_SIZE
)

plot_3d_trajectory(y_pred_real, y_true_real, forecast_step=0)
