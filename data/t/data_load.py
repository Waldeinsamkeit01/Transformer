# -*- coding:utf-8 -*- 
# coding:unicode_escape
# @Author: Lemon00
# @Time: 2023/8/16 14:10
# @File: data_load
import torchtext.datasets as datasets
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split

import pandas as pd

# dataloader = DataLoader("../example_10M.csv",batch_size=32,shuffle=True)
# dataset = pd.read_csv("../example_1k.csv")
#
# train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
# train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
# train_data.to_csv("train.csv", index=False)
# test_data.to_csv("test.csv",index=False)
# val_data.to_csv("val.csv", index=False)
#
# dataset.TabularDataset(
#
# )


# train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
# test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)


# train_iter, valid_iter, test_iter = datasets.Multi30k(
#     language_pair=("de", "en")
# )
# print("")


from torch.utils.data import Dataset
import pandas as pd  # 这个包用来读取CSV数据


# 继承Dataset，定义自己的数据集类 mydataset
from torchtext.data import to_map_style_dataset


class mydataset(Dataset):
    def __init__(self, csv_file):  # self 参数必须，其他参数及其形式随程序需要而不同，比如(self,*inputs)
        self.csv_data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = self.csv_data.values[idx]
        return data


data_iter = mydataset('train.csv')
data_iter_map = to_map_style_dataset(data_iter)
data_sampler = (
    DistributedSampler(data_iter_map) if True else None
)
print("data[0]")

