class Config:
    task_name = "long_term_forecast"
    seq_len = 96
    pred_len = 96
    enc_in = 3   # number of variables
    d_model = 128
    n_heads = 8
    e_layers = 2
    d_ff = 512
    dropout = 0.1
    factor = 1
    activation = "gelu"
    e_layers = 2 
    patch_len = 16
    stride = 8
