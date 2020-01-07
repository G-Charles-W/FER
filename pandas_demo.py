import pandas as pd

row_train_acc = []
row_train_los = []
row_val_acc = []
row_val_los = []

for i in range(10):
    row_train_acc.append(i)
    row_train_los.append(i * i)
    row_val_acc.append(i * 10)
    row_val_los.append(i * 5)

a = [row_train_acc, row_train_los, row_val_acc, row_val_los]

df = pd.DataFrame(a, index=['train_acc', 'train_loss', 'val_acc', 'val_loss'], columns=[f'epoch_{j}' for j in range(10)])

df.to_csv('traning_record.csv', float_format='%.2f', na_rep="NAN!")

print(df)
