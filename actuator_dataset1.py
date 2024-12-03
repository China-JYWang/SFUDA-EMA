# import os
import torch
import numpy as np
import pandas as pd
# from torchvision import datasets, transforms
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sequence_aug import *

class dataset(data.Dataset):
    def __init__(self, list_data):

        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        self.transform = Compose([
                Reshape()])
        self.classes = np.unique(self.labels)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        seq = self.transform(seq)
        label = self.labels[item]
        return seq, label, item

def get_dataset(rootdir,batch_size,randomstate):
    print(rootdir)
    source_data = np.load(rootdir + '/' + 'data_features_source.npy').astype(np.float32)
    source_label = np.load(rootdir + '/' + 'data_labels_source.npy').astype(np.uint8)
    # obtain the source domain dataset
    source_data_list = []
    for each in source_data:
        source_data_list.append(each.reshape(-1, 1))

    data_pd = pd.DataFrame({'data': source_data_list, 'label': source_label.tolist()})
    print('总的样本数为：', len(data_pd['label']))
    train_pd, val_pd = train_test_split(data_pd, train_size=0.5, random_state=randomstate, stratify=data_pd["label"])
    source_train = dataset(list_data=train_pd)
    source_val = dataset(list_data=val_pd)
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    source_train_loader = torch.utils.data.DataLoader(source_train, batch_size=batch_size, shuffle=True, **kwargs)
    source_val_loader = torch.utils.data.DataLoader(source_val, batch_size=batch_size, shuffle=True, **kwargs)

    return source_train_loader,source_val_loader,source_train.classes


def get_dataset_all(rootdir, batch_size, randomstate):
    # print('2')
    print(rootdir)
    source_data = np.load(rootdir + '/' + 'all_data_set.npy').astype(np.float32)
    # print('1')
    source_label = np.load(rootdir + '/' + 'all_data_label.npy').astype(np.uint8)
    # obtain the source domain dataset
    source_data_list = []
    for each in source_data:
        source_data_list.append(each.reshape(-1, 1))

    data_pd = pd.DataFrame({'data': source_data_list, 'label': source_label.tolist()})
    # print('总的样本数为：', len(data_pd['label']))
    train_pd, val_pd = train_test_split(data_pd, train_size=0.5, random_state=randomstate, stratify=data_pd["label"])
    # print(train_pd)
    # 不平衡数据
    # train_0 = train_pd[train_pd.label == 0][:300]
    # train_1 = train_pd[train_pd.label == 1][:100]
    # train_2 = train_pd[train_pd.label == 2][:100]
    # train_3 = train_pd[train_pd.label == 3][:100]
    # train_pd = pd.concat([train_0, train_1, train_2, train_3])

    print(len(train_pd))
    source_train = dataset(list_data=train_pd)
    source_val = dataset(list_data=val_pd)
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    source_train_loader = torch.utils.data.DataLoader(source_train, batch_size=batch_size, shuffle=True, **kwargs)
    source_val_loader = torch.utils.data.DataLoader(source_val, batch_size=batch_size, shuffle=True, **kwargs)

    return source_train_loader,source_val_loader,source_train.classes
