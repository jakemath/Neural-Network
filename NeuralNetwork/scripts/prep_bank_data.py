#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Split data


print('Cleaning dataset...')
bank = pd.read_csv('/NeuralNetwork/NeuralNetwork/data/banking.csv')
print(bank.head())
bank = bank.replace('unknown', np.NaN)
bank = bank.dropna(axis=0)
bank['education'] = bank['education'].map(
    lambda x: 'basic' if x in ['basic.4y', 'basic.6y', 'basic.9y'] else x)
bank['job'] = bank['job'].map(
    lambda x: 'white-collar' if x in ['admin', 'management', 'entrepreneur', 'technician']
                             else 'blue-collar' if x in ['services, housemaid']
                             else x)
bank.loc[bank['job'] == 'admin', 'job'] = 'white-collar' # Group white-collar jobs together
bank.loc[bank['job'] == 'management', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'entrepreneur', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'technician', 'job'] = 'white-collar'
bank.loc[bank['job'] == 'services', 'job'] = 'blue-collar' # Group blue-collar/service jobs together
bank.loc[bank['job'] == 'housemaid', 'job'] = 'blue-collar'
category_cols = [col for col in bank.columns if bank[col].dtype != 'float64' and col != 'y']
for col in category_cols:
    bank[col] = bank[col].map(dict(zip(sorted(bank[col].unique()), range(len(bank[col].unique())))))
float_cols = [col for col in bank.columns if bank[col].dtype == 'float64']
for col in float_cols:
    bank[col] = (bank[col] - bank[col].mean())/bank[col].std()
print(bank.head())
train, test = train_test_split(bank, test_size=0.3)  # Split data, 30% for testing
train.to_csv('/NeuralNetwork/NeuralNetwork/data/train_bank.txt', header=None, index=False)
test.to_csv('/NeuralNetwork/NeuralNetwork/data/test_bank.txt', header=None, index=False)

