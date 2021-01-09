#! Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print('Cleaning dataset...')
bank = pd.read_csv('../data/banking.csv')
print(bank.head())
bank = bank.replace('unknown', np.NaN)
bank = bank.dropna(axis=0)
print('Classifying occupations...')
bank.loc[bank['education'].isin(['basic.4y', 'basic.6y', 'basic.9y']), 'education'] = 'basic'
bank.loc[bank['job'].isin(['admin', 'management', 'entrepreneur', 'technician']), 'job'] = 'white-collar'
bank.loc[bank['job'].isin(['services, housemaid']), 'job'] = 'blue-collar'
print('Encoding categories...')
category_cols = [col for col in bank.columns if bank[col].dtype != 'float64' and col != 'y']
for col in category_cols:
    bank[col] = bank[col].map(dict(zip(sorted(bank[col].unique()), range(len(bank[col].unique())))))
print('Normalizing numerical values...')
float_cols = [col for col in bank.columns if bank[col].dtype == 'float64']
for col in float_cols:
    bank[col] = (bank[col] - bank[col].mean())/bank[col].std()
print(bank.head())
train, test = train_test_split(bank, test_size=0.3)  # Split data - 30% for testing
train.to_csv('../data/train_bank.txt', header=None, index=False)
test.to_csv('../data/test_bank.txt', header=None, index=False)

