class Config:
    task_name = "long_term_forecast" #default
    seq_len = 96    #lookback window
    pred_len = 96   #forecast length
    enc_in = 3      #number of variables in the data
    d_model = 128   #dimension of the model 
    n_heads = 4     
    e_layers = 2    #how many blocks 
    d_ff = 512      #inner dimension
    dropout = 0.1  
    factor = 1
    activation = "gelu" #activation function
    patch_len = 16  #patch length
    stride = 8      #window for patching

    #training settings 
    batch_size = 32
    epochs = 200
    patience = 20
    lr = 0.0001
