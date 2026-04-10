class Config:
    task_name = "long_term_forecast"
    seq_len = 96
    pred_len = 96
    enc_in = 3   # number of variables
    d_model = 128
    patch_num = 12
    e_layers = 2
    d_ff = 512
    dropout = 0.1
    e_layers = 2 
    patch_len = 16
    stride = 8
    head_dropout = 0.0
    bias = True
    avg = True
    max = True
    dt_rank = 8
    d_ff = 256
    dt_init = "random"
    d_state = 16
    
