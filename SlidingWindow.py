class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_length, forecast_length):
        """
        data: numpy array (T, V)
        returns:
            X: (V, input_length)
            y: (V, forecast_length)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_length = input_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data) - self.input_length - self.forecast_length + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_length]              # (L, V)
        y = self.data[idx + self.input_length :
                      idx + self.input_length + self.forecast_length]

        x = x.T  # (V, L)
        y = y.T  # (V, F)

        return x, y
