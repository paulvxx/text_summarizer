import csv
import os
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)  
print(data_dir)

files = [('sequences100_test.csv',''), ('sequences200_validate.csv',''), ('sequences500_test.csv',''), 
        ('sequences1000_train.csv',''), ('sequences1000_validate.csv',''), ('sequences5000_train.csv','')]

train_X = []

for filename in files:
    file_path = os.path.join(data_dir, filename)
    print(file_path)
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
            a = np.array([int(i) for i in row])
            # sequence of 64 numbers
            train_X.append(a[:64])
            # 5 elements with greatest frequency then ascending
            a[64:]
        train_X = np.array(train_X)

