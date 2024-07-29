# import final dataset into main file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

originalDataset = pd.read_csv('S3T3CPPT/final_hateXplain.csv')
originalDataset = originalDataset[originalDataset['Sexual Orientation'] == 'Homosexual']
originalDataset.to_csv('out.csv')