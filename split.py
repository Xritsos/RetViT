import pandas as pd
import shutil


if __name__ == "__main__":
    
    df_val = pd.read_csv('./data/val.csv')
    df_test = pd.read_csv('./data/test.csv')
    
    val_names = df_val['Fundus']
    test_names = df_test['Fundus']
    
    for image in val_names:
        try:
            shutil.move(f'./data/train/{image}', f'./data/val/{image}')
        except Exception:
            pass
    
    for image in test_names:
        try:
            shutil.move(f'.data/train/{image}', f'./data/test/{image}')
        except Exception:
            pass
    