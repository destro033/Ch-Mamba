import joblib
import torch

from model import Model
from arguments import Config
from evaluation import (
    load_test_dataframe,
    evaluate_flights_full
)


# CHANGE THESE IF NEEDED
CSV_PATH = "Drone Onboard Multi-Modal Feature-Based Visual Odometry Dataset.csv"
WEIGHTS_PATH = "mamba_best_model.pth"
SCALER_PATH = "scaler_mamba.pkl"

configs = Config()

INPUT_LENGTH = configs.
FORECAST_LENGTH = configs.
BATCH_SIZE = configs.batch_size

FLIGHT_1 = 17
FLIGHT_2 = 18


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# Load dataframe
df = load_test_dataframe(CSV_PATH, sep=";")
feature_cols = ['position_x', 'position_y', 'position_z']

# Load model

model = Model(configs).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()


# Load scaler
scaler = joblib.load(SCALER_PATH)

results = evaluate_flights_full(
    model=model,
    df=df,
    flight_ids=[FLIGHT_1, FLIGHT_2],
    feature_cols=feature_cols,
    scaler=scaler,
    device=device,
    input_length=INPUT_LENGTH,
    forecast_length=FORECAST_LENGTH,
    batch_size=BATCH_SIZE,
    save_path="cmamba_results.npz"
)


