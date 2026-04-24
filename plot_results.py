

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


MODEL_A_PATH = "cmamba_results.npz"
MODEL_B_PATH = "patchtst_results.npz"

MODEL_A_NAME = "cmamba"
MODEL_B_NAME = "patchtst"


# =========================
# Load saved results
# =========================
a = np.load(MODEL_A_PATH)
b = np.load(MODEL_B_PATH)

mae_a = a["mae_xyz"]
mae_b = b["mae_xyz"]

error_a = a["error_per_forecast_step"]
error_b = b["error_per_forecast_step"]

pred_a = a["preds"]
gt_a = a["gt"]

pred_b = b["preds"]
gt_b = b["gt"]

counts = a["flight_sample_counts"]
flight_ids = a["flight_ids"]


# =========================
# 1. MAE X/Y/Z comparison
# =========================
labels = ["X", "Y", "Z"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 4))
plt.bar(x - width / 2, mae_a, width, label=MODEL_A_NAME)
plt.bar(x + width / 2, mae_b, width, label=MODEL_B_NAME)

plt.xticks(x, labels)
plt.ylabel("MAE (m)")
plt.title("MAE Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("mae_comparison.pdf", bbox_inches="tight")


# =========================
# 2. Euclidean error over forecast steps
# =========================
steps = np.arange(1, len(error_a) + 1)

plt.figure(figsize=(8, 4))
plt.plot(steps, error_a, label=MODEL_A_NAME)
plt.plot(steps, error_b, label=MODEL_B_NAME)

plt.xlabel("Forecast Step")
plt.ylabel("Euclidean Error (m)")
plt.title("Euclidean Error Across Forecast Horizon")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("euclidean_error_curve.pdf", bbox_inches="tight")


# =========================
# Helper: split flights
# =========================
def split_by_flight(array, counts):
    flights = []
    start = 0

    for count in counts:
        end = start + count
        flights.append(array[start:end])
        start = end

    return flights


pred_a_flights = split_by_flight(pred_a, counts)
gt_a_flights = split_by_flight(gt_a, counts)

pred_b_flights = split_by_flight(pred_b, counts)
gt_b_flights = split_by_flight(gt_b, counts)


# =========================
# 3. 3D trajectory plots
# =========================
def plot_3d_flight(
    gt,
    pred_model_a,
    pred_model_b,
    flight_id,
    forecast_step=0
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_idx, y_idx, z_idx = 0, 1, 2

    ax.plot(
        gt[:, forecast_step, x_idx],
        gt[:, forecast_step, y_idx],
        gt[:, forecast_step, z_idx],
        label="Ground Truth",
        linewidth=2
    )

    ax.plot(
        pred_model_a[:, forecast_step, x_idx],
        pred_model_a[:, forecast_step, y_idx],
        pred_model_a[:, forecast_step, z_idx],
        label=MODEL_A_NAME
    )

    ax.plot(
        pred_model_b[:, forecast_step, x_idx],
        pred_model_b[:, forecast_step, y_idx],
        pred_model_b[:, forecast_step, z_idx],
        label=MODEL_B_NAME
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Trajectory Comparison - Flight {flight_id}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"trajectory_flight_{flight_id}.pdf", bbox_inches="tight")


for i, flight_id in enumerate(flight_ids):
    plot_3d_flight(
        gt=gt_a_flights[i],
        pred_model_a=pred_a_flights[i],
        pred_model_b=pred_b_flights[i],
        flight_id=flight_id,
        forecast_step=0
    )


plt.show()
