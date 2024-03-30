import pandas as pd
import shutil


if __name__ == "__main__":
    
    df_val = pd.read_csv('./data/val.csv')
    df_test = pd.read_csv('./data/test.csv')
    df_train = pd.read_csv('./data/train.csv')
    
    val_names = df_val['Fundus']
    test_names = df_test['Fundus']
    train_names = df_train['Fundus']
    
    for image in val_names:
        try:
            shutil.move(f'./data/dataset/{image}', f'./data/val/{image}')
        except Exception as ex:
            print(ex)
    
    for image in test_names:
        try:
            shutil.move(f'./data/dataset/{image}', f'./data/test/{image}')
        except Exception as ex:
            print(ex)
        
    for image in train_names:
        try:
            shutil.move(f'./data/dataset/{image}', f'./data/train/{image}')
        except Exception as ex:
            print(ex)
    