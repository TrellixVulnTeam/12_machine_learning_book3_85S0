import tarfile

#%%
with tarfile.open(
    '/Users/rukaoide/Downloads/aclImdb_v1.tar', 'r') as tar:
    tar.extractall()


#%%
import pyprind
import pandas as pd
import os

basepath = '/Users/rukaoide/Downloads/aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']
