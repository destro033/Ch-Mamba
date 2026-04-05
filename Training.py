train_losses = []
val_losses = []

# Initialize model, loss function, and optimizer
model = CMamba(args).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 200

import copy

patience = 20       # number of epochs to wait for early stopping
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict()) # store the best weights found


# Training of the model
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

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}")


        # Put model on evaluation mode, meaning no more updates on weights
        # To check how it behaves in new data (validation set)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (X_batch, y_batch) in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()


        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

        # Early Stopping Logic

        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model_wts = copy.deepcopy(model.state_dict())
          epochs_no_improve = 0
          print(" Validation loss improved — saving best model")
        else:
          epochs_no_improve += 1

        if epochs_no_improve >= patience:
          break

# Store the best result (best validation loss) from all the epochs
model.load_state_dict(best_model_wts)
print(f" Best model restored (val loss = {best_val_loss:.4f})")

