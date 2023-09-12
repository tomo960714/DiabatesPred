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

def preprocess_data(args):
    """
    Preprocess the data
    # TODO: this function should be implemented
    # normalize the data
    """

    # Load the csv file
    df = load_csv(args)

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