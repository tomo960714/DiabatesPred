# Imports
import os
import argparse


# 3rd party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neptune.new as neptune
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# local imports
import utils as ut
from model import GenderNetwork
from dataset import GenderDataset


def main(args):

    # load dataset:
    gender_dataframe = ut.load_csv(args)

    #preprocess df
    gender_dataframe = ut.preprocess_data(args,gender_dataframe)

    #data loader
    dataset = GenderDataset(gender_dataframe,args)

    #split dataset
    args.train_dataset, args.val_dataset, args.test_dataset = ut.split_dataset(dataset,args)


    #build model
    model = GenderNetwork()
    model = model.int()

    # handler device selection
    if args.device_selector == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    model.to(args.device)


    # train model


    #validate model

    #visualize results
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='ptbxl_database.csv')
    parser.add_argument('--data_path', type=str, default='\data\ptb-xl')
    parser.add_argument('--used_cols', type=list, default=['ecg_id','weight','height','sex'], help='columns to use from the csv file')
    parser.add_argument('--test_type',type=str, default='gender')
    parser.add_argument('--train_ratio',type=float,default=0.8,help='ratio of train data, default is 0.8, train,validation and test ratio should sum to 1')
    parser.add_argument('--val_ratio',type=float,default=0.1,help='ratio of validation data, default is 0.1, train,validation and test ratio should sum to 1')
    parser.add_argument('--test_ratio',type=float,default=0.1,help='ratio of test data, default is 0.1, train,validation and test ratio should sum to 1')
    parser.add_argument('--sampling_rate',type=int,default=100,help='sampling rate of the data, default is 100')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size, default is 64')
    parser.add_argument('--epochs',type=int,default=100,help='number of epochs, default is 100')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate, default is 0.001')
    parser.add_argument('--seed',type=int,default=42,help='seed for reproducibility, default is 42')
    parser.add_argument('--device_selector',type=str,default='cuda',help='cuda or cpu, default is cuda')
    parser.add_argument('--loss',type=str,default='CrossEntropyLoss',help='loss function, default is CrossEntropyLoss, if different implement it in ttv.py/train_model before using it')
    parser.add_argument('--max_length',type=int,default=0,help='Limiter for testing. Limits the max length of the data, 0 means no limiter. Default is 0')


    
    args = parser.parse_args()
    main(args)