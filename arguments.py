class Config:
    task_name = "long_term_forecast" #default
    seq_len = 96    #input length (look back window)
    pred_len = 96   #forecast horizon length
    enc_in = 3      #number of variables in the data
    d_model = 128   #dimension of the model 
    patch_num = 12  #how many patches to create
    e_layers = 2    #how many c-mamba blocks
    d_ff = 256      #inner dimension of the model 
    dropout = 0.1 
    patch_len = 16  #length of each patch 
    stride = 8      #window for patching
    head_dropout = 0.0 #disable neurons
    bias = True
    avg = True
    max = True
    dt_rank = 8
    dt_init = "random"
    d_state = 16
    dt_max = 0.1
    dt_min = 0.001
    dt_init_floor = 1e-4
    dt_scale = 1.0
    d_state = 16    #dimension of state space
    c_out = 3       #have to be the same as enc_in
    reduction = 2
    gddmlp = True   #whether or not to use channel attention
    pscan = True    #whether or not to use parallel scanning for faster computation
    sigma = 0.1     #value of sigma
    use_channel_mixup = True #whether or not to use channel mixup

    #training settings 
    batch_size = 32
    epochs = 200
    patience = 20
    lr = 0.0001   #learning rate
    
