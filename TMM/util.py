import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset
import os
import logging
import time
import csv
import codecs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import sklearn.metrics
import torch_geometric
from scipy.sparse import coo_matrix

from sklearn.metrics import auc, f1_score, roc_curve, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDatasetWithAdj(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = torch.FloatTensor(sample[1:-1].astype(float))  # 将特征转换为float类型的张量
        label = sample[-1]
        return features, label



def load_adj_data(folder_path,x):
    # 获取目录下所有文件名
    file_names = os.listdir(folder_path)
    # 初始化一个空字典，用于存储数据，以文件名作为键，数据作为值
    adj_dict = {}
    # 遍历文件名列表，加载数据并添加到字典中
    for file_name in file_names:
        # 根据文件名构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)
        # 加载CSV文件数据，
        adj_data = pd.read_csv(file_path, header=None).values.astype(float)
        adj_data = torch.LongTensor(np.where(adj_data > x, 1, 0))
        # 获取不带后缀的文件名
        file_name_without_extension = os.path.splitext(file_name)[0]
        # 将数据添加到字典中，以不带后缀的文件名作为键
        adj_dict[file_name_without_extension] = adj_data
    return adj_dict



def specificity_score(y_true, y_pred):
    # 计算SPE
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spe
def sensitivity_score(y_true, y_pred):
    # 计算SEN
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return sen

################
# Layer Utils
################
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

################
# subgraph
################
def adj_to_PyG_edge_index(adj):
    coo_A = coo_matrix(adj)
    edge_index, edge_weight = torch_geometric.utils.convert.from_scipy_sparse_matrix(coo_A)
    return edge_index

def data_to_PyG_data(x, edge_index, y):
    out_data = x
    out_edge_index = edge_index
    out_label = y
    PyG_data = torch_geometric.data.Data(x=out_data, edge_index=out_edge_index, y=out_label)
    return PyG_data

def PyG_edge_index_to_adj(edge_index):
    adj = torch_geometric.utils.to_dense_adj(edge_index=edge_index)
    return adj

def data_write_csv(file_name, datas):
  file_csv = codecs.open(file_name,'w+','utf-8')
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("doc saved")