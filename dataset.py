
#%%
import pandas as pd
import os
from scipy.fft import fft
from utils import load_single_raw_data, calc_BMI,reset_sex
from constants import REC_PATH, SAMPLING_RATE,DATASET_LIMIT
from torchvision import transforms

#%%
class DataSet():
    def __init__(self,args,input_df,transforms):
        """
        Custom dataset class for PTB-XL dataset
        Args:
        - input_df: the dataset as a dataframe object


        """
        self.data = input_df
        self.transforms = transforms
    def __len__(self):
        return len(self.data['ecg_id'])
    
    def __getitem__(self,idx,args):
        
        # get label
        if args.test_type.lower() == 'gender':
            label = self.data['sex'].iloc[idx]
        elif args.test_type.lower() == 'bmi':
            label = self.data['BMI'].iloc[idx]
        else:
            raise ValueError(f'Invalid test, please choose between gender or BMI, {args.test_type} is invalid')
        
        #get data
        rec = load_single_raw_data(self.data['filename_lr'].iloc[idx],args.sampling_rate,args.data_path)

        #transform loaded recording to tensor
        rec_as_tensor = self.transform(rec.astype(float))

        return rec_as_tensor,label