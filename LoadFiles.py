import torch
#change name here if needed
args_dict = torch.load("args.pth", map_location="cpu")

# REMOVE internal-only keys, they are fixed anyway in the program, just with different names so to avoid errors we do this.
args_dict.pop("v", None)
args_dict.pop("num_patches", None)
args_dict.pop("d_inner", None)

args = ModelArgs(**args_dict)

model = CMamba(args)
#change name here if needed 
model.load_state_dict(
    torch.load("weights.pth", map_location="cpu")
)
model.eval()

import joblib
#change name here if needed
scaler = joblib.load("scaler.pkl")

