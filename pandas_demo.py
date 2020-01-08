import pandas as pd
import os
import numpy as np


def write_2_csv(name):
    df = pd.DataFrame(columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
    df.to_csv(name, float_format='%.2f', na_rep="NAN!", index=False)
    for i in range(10):
        row_train_acc = i
        row_train_los = i * i
        row_val_acc = i * 10
        row_val_los = i * 5
        dict_ = {'train_acc': row_train_acc,
                 'train_loss': row_train_los,
                 'val_acc': row_val_acc,
                 'val_loss': row_val_los, }
        df = pd.DataFrame(dict_, index=[i])
        df.to_csv(name, mode='a', header=False)


def read_csv(name):
    df = pd.read_csv(name)
    print(df)


write_2_csv('my_csv.csv')
read_csv('my_csv.csv')
