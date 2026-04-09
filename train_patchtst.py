import copy
import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing_patchtst import create_dataloaders_from_csv
from model_patchtst import Model
from arguments_patchtst import Config
import pandas as pd
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

train_loader, val_loader, scaler = create_dataloaders_from_csv(
    csv_path="Drone Onboard Multi-Modal Feature-Based Visual Odometry Dataset.csv",
    input_length=configs.seq_len,
    forecast_length=configs.pred_len,
    batch_size=32
)

train_losses = []
val_losses = []

configs = Config()

model = Model(configs).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 200

patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch [{epoch+1}/{epochs}]")

    for (X_batch, y_batch) in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (X_batch, y_batch) in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        print("Validation loss improved — saving best model")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        break

model.load_state_dict(best_model_wts)
print(f"Best model restored (val loss = {best_val_loss:.4f})")

#save the weights of the best model
torch.save(model.state_dict(), "patchtst_best_model.pth")
print("Model weights saved to patchtst_best_model.pth")

#save scaler
joblib.dump(scaler, "scaler_patchtst.pkl")
print("Scaler saved to scaler_patchtst.pkl")

#save losses into a csv file
results_df = pd.DataFrame({
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "best_val_loss": [best_val_loss] * len(train_losses)  # repeated value
})

results_df.to_csv("training_results_patchtst.csv", index=False)

print("Results saved to training_results_patchtst.csv")
