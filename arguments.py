from model import ModelArgs

args = ModelArgs(
        d_model=128,          # Dimension of the model
        n_layer=2,            # Number of C-Mamba blocks
        seq_len=96,           # Length of input sequence (look-back window)
        num_channels=3,       # Number of numerical channels in data (how many variables (x,y,z))
        patch_len=16,         # Length of each patch
        stride=8,             # Stride for patching
        forecast_len=96,      # Number of future time steps to predict
        d_state=16,           # Dimension of SSM state
        expand=2,             # Expansion factor for inner dimension
        dt_rank='auto',       # Rank for delta projection, 'auto' sets it to d_model/16
        d_conv=4,             # Kernel size for temporal convolution
        pad_multiple=8,       # Padding to ensure sequence length is divisible by this
        conv_bias=True,       # Whether to use bias in convolution
        bias=False,           # Whether to use bias in linear layers
        sigma=0.1,            # Standard deviation for channel mixup
        reduction_ratio=2,    # Reduction ratio for channel attention
        verbose=False         # Whether to make the model print analytically the tensor shapes changes 
)
