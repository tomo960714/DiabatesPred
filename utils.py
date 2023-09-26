#%%
import pandas as pd
import numpy as np
import os
import argparse
def load_csv(args):
    """
    Load the csv file and return a pandas dataframe
    """
    
    df = pd.read_csv(os.path.join(args.data_path, args.csv_path), header=None, usecols=args.used_cols)
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