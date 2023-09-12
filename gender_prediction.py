import argparse


def main(args):
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='ptbxl_database.csv')
    parser.add_argument('--data_path', type=str, default='\data\ptb-xl')
    parser.add_argument('--used_cols', type=list, default=['ecg_id','weight','height','sex'])
    
    args = parser.parse_args()
    main(args)