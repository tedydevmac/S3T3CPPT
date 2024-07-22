# import final dataset into main file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

originalElecStoreDataset = pd.read_csv('S3T3CPPT/datasets/kz.csv')
elecStoreDataset = pd.DataFrame()
customer_id = []
elecStoreDatasetData = {
    'user_id': [],
    'order_id': [],
    'event_time': [],
    'order_id': [],
    'product_id': [],
    'category_id': [],
    'category_code': [],
    'brand': [],
    'price': [],
}

# shoppingTrendsDataset = pd.read_csv('S3T3CPPT/shopping_trends_updated.csv')

print(originalElecStoreDataset.head())
# print(shoppingTrendsDataset)

k = 0
for idx, row in originalElecStoreDataset.iterrows():
    if row['user_id'] in elecStoreDataset:
        for column in row:
            newValue = row.loc[:,column].item()
    else:
        i = 0
        for itm in row:
            print(elecStoreDatasetData[row.index[i]])
            elecStoreDatasetData[row.index[i]] = elecStoreDatasetData[row.index[i]][0:(elecStoreDatasetData['user_id'].index(itm))]+elecStoreDatasetData[row.index[i]][0:(elecStoreDatasetData['user_id'].index(itm))]+elecStoreDatasetData[row.index[i]][0:(elecStoreDatasetData['user_id'].index(itm))]
            i += 1
        # newRow = (row.to_frame().T).set_index(pd.Index([row['user_id']]))
        # print(newRow)
        # elecStoreDataset = pd.concat([elecStoreDataset,newRow.iloc[:,:-1]])
    k += 1
    if k > 10:
        break
print(elecStoreDataset)
