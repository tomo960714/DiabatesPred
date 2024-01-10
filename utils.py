#%%
import pandas as pd
import numpy as np
import os
import argparse
import torch
def load_csv(args):
    """
    Load the csv file and return a pandas dataframe
    """
    
    df = pd.read_csv(os.path.join(args.data_path, args.csv_path), header=None, usecols=args.used_cols)
    
    # limit the dataset length
    if args.dataset_length == 0:
        pass
    else:
        df = df[:args.dataset_length]
    return df

def calculate__BMI(df):
    """
    Calculate the BMI from the weight and height
    """
    df['BMI'] = df['weight'] / (df['height']/100)**2
    return df

def preprocess_data(args,df):
    """
    Preprocess the data
    # TODO: this function should be implemented
    # normalize the data
    """

    # Limit the dataset length
    if args.dataset_length == 0:
        pass
    else:
        df = df[:args.dataset_length]
    

    # Fix the data types
    df['weight'] = df['weight'].astype(float)
    df['height'] = df['height'].astype(float)
    
    # drop the NaN values
    tmp_df = df.drop(df[df['weight'].isna() | df['height'].isna() |df['sex'.isna() ]].index , inplace=True)    
    
    # Calculate the BMI
    df = calculate__BMI(df)

    return df

#TODO: Padding function should be implemented
def padding_data(args,df):
    """
    Pad the data to have the same length
    """

    padding_data = np.zeros((args.dataset_length, args.max_length))
    for i in range(args.dataset_length):
        padding_data[i,:df[i].shape[0]] = df[i]
    return padding_data

def load_single_raw_data(filename, sampling_rate, path):
    data = wfdb.rdsamp(path+filename) 
    #print(data)
    #print('data:',data.shape)
    data_arr = np.asarray(data[0]).transpose()
    #print(data_arr.shape)
    #print('data_arr',data_arr.shape)
    return data_arr

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    
    data = np.array([signal for signal, meta in data])
    #print(data)
    return data

# split dataset
def split_dataset(dataset,args):
    """
    Split the dataset into train, test and validation
    """

    #check for correct ratios
    if args.train_ratio + args.val_ratio + args.test_ratio != 1:
        raise ValueError('Train, validation and test ratio should sum to 1')
    
    # get dataset length
    dataset_length = len(dataset)

    # get indices
    # indices = list(range(dataset_length))
    # np.random.shuffle(indices)

    # split dataset into train, validation and test with torch.utils.data.random_split
    train_length = int(np.floor(args.train_ratio * dataset_length))
    test_length = int(np.floor(args.test_ratio * dataset_length))
    val_length = dataset_length - train_length - test_length
    train_dataset,test_dataset ,val_dataset  = torch.utils.data.random_split(dataset, [train_length, test_length,val_length])
    args.set_sizes = {'train':train_length, 'test':test_length,'validation':val_length}
    return train_dataset, val_dataset, test_dataset
