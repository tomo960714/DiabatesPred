# Imports
import os
import argparse

# 3rd party imports
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import neptune.new as neptune
from tqdm import tqdm,trange


# local imports

# Path: ttv.py
# Functions for train, test and validate model
#%%

def model_loop(model,args):
    """
    Train the model
    """
    # handle arguments
    device = args.device
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    seed = args.seed

    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # define loss function and optimizer
    # check if args.loss is a valid loss function
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Unimplemented loss function, please implement it in ttv.py/train_model before using it')
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # define data loader
    train_loader = torch.utils.data.DataLoader(args.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(args.val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(args.test_dataset, batch_size=batch_size, shuffle=True)

    args.dataloaders = {'train':train_loader, 'test':test_loader,'validation':val_loader}

    # train epoch loop with tqdm
    logger = trange(args.epochs,desc =f"Epoch:0, Loss:0")
    for epoch in logger:


    logger.set_description(f"Epoch:{epoch}, Loss:{loss:.4f}")
    args.writer[''].append(loss)