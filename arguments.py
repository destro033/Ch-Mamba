class Config:
    task_name = "long_term_forecast" #default
    seq_len = 96    #input length (look back window)
    pred_len = 96   #forecast horizon length
    enc_in = 3      #number of variables in the data
    d_model = 128   #dimension of the model (patch embedding)
    patch_num = 12  #how many patches to create
    e_layers = 2    #how many c-mamba blocks
    d_ff = 256      #dimension of linear projection in Mamba
    dropout = 0.1   #dropout in C-Mamba block
    head_dropout = 0.0 #dropout before the final linear projection layer 
    bias = True  #default
    avg = True   #default
    max = True   #default
    dt_rank = 8  #default
    dt_init = "random"  #default
    dt_max = 0.1   #default
    dt_min = 0.001 #default
    dt_init_floor = 1e-4  #default
    dt_scale = 1.0   #default
    d_state = 16    #dimension of state space
    c_out = 3       #have to be the same as enc_in
    reduction = 2   #reduction rate for GDD-MLP
    gddmlp = True   #whether or not to use channel attention
    pscan = True    #whether or not to use parallel scanning for faster computation in training
    sigma = 0.1     #standard deviation for Channel Mixup
    use_channel_mixup = True #whether or not to use channel mixup

    #training settings 
    batch_size = 32
    epochs = 200
    patience = 20 #how many epochs without improvement in validation loss to wait
    lr = 0.0001   #learning rate
    
