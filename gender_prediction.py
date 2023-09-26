# Imports
import os
import argparse


# 3rd party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neptune.new as neptune

# local imports
import utils as ut

def main(args):

    # load dataset:
    df = ut.load_csv(args)

    #preprocess df
    df = ut.preprocess_data(args,df)

    #data loader

    #build model

    # train model

    #validate model

    #visualize results
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='ptbxl_database.csv')
    parser.add_argument('--data_path', type=str, default='\data\ptb-xl')
    parser.add_argument('--used_cols', type=list, default=['ecg_id','weight','height','sex'])
    parser.add_argument('--test_type',type=str, default='gender')
    parser.add_argument('--sampling_rate',type=int,default=100)
    parser.add_argument('--')
    
    args = parser.parse_args()
    main(args)