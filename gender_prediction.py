# Imports
import os
import argparse


# 3rd party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neptune.new as neptune
import math
from torchvision import transforms,DataLoader
import torch


# local imports
import utils as ut
from dataset import DataSet

def main(args):

    # load dataset:
    loaded_data = ut.load_csv(args)

    # preprocess df
    loaded_data = ut.preprocess_data(args,loaded_data)

    # dataset
    dataset_transforms = {transforms.ToTensor()}
    loaded_Dataset = DataSet(loaded_data,dataset_transforms)

    # Split data
    train_length = math.floor(loaded_Dataset.__len__()* args.train_length)
    test_length =loaded_Dataset.__len__()-train_length

    train_dataset,test_dataset =torch.utils.data.random_split(loaded_Dataset,[train_length,test_length])
    
    # data loader
    train_dataloader = DataLoader(train_dataset)
    test_dataloader = DataLoader(test_dataset)

    # save variables
    args.dataloaders = {
        'Train':
            train_dataloader,
        'Test':
            test_dataloader,
    }
    args.set_sizes = {
        'Train':
            train_length,
        'Test':
            test_length,
    }

    # get device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # build model

    # train model

    # validate model

    # visualize results
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--csv_path', type=str, default='ptbxl_database.csv')
    parser.add_argument('--data_path', type=str, default='\data\ptb-xl')
    
    parser.add_argument('--name',type=str,default='test')

    # Dataset parameters
    parser.add_argument('--used_cols', type=list, default=['ecg_id','weight','height','sex'])
    parser.add_argument('--test_type',type=str, default='gender')
    parser.add_argument('--sampling_rate',type=int,default=100)
    parser.add_argument('--train_size',type=float,default=0.75)
    
    # model parameters
    #TODO: find default values:
    parser.add_argument('--model_num_layers',type=int, default=)
    parser.add_argument('--model_hidden_dim',type=int,default=)
    parser.add_argument('--model_input_dim',type=int,default=)
    
    parser.add_argument('--batch_size',type=int,default=64)
    
    args = parser.parse_args()
    main(args)