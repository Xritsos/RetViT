import pandas as pd
import shutil


if __name__ == "__main__":
    
    df_val = pd.read_csv('../csv_backup/N-D-C-M/val.csv')
    df_test = pd.read_csv('../csv_backup/N-D-C-M/test.csv')
    df_train = pd.read_csv('../csv_backup/N-D-C-M/train.csv')
    
    val_names = df_val['Fundus']
    test_names = df_test['Fundus']
    train_names = df_train['Fundus']
    
    for image in val_names:
        try:
            shutil.move(f'./data/dataset/{image}', f'data/ODIR/val/{image}')
        except Exception as ex:
            print(ex)
    
    for image in test_names:
        try:
            shutil.move(f'./data/dataset/{image}', f'data/ODIR/test/{image}')
        except Exception as ex:
            print(ex)
        
    for image in train_names:
        try:
            shutil.move(f'./data/dataset/{image}', f'data/ODIR/train/{image}')
        except Exception as ex:
            print(ex)
