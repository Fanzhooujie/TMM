from model import *
from sklearn.metrics import roc_auc_score, f1_score,accuracy_score
import pandas as pd
import pdb
from tqdm import tqdm
import shap
import csv
import random
from util import *
import matplotlib
import os
matplotlib.use('Agg')

adj_AHBA = np.array(pd.read_csv('./NCvsAD/AHBA.csv', header=None))[:, :].astype(float)
adj_AHBA = torch.LongTensor(np.where(adj_AHBA > args.adj_AHBA, 1, 0))
adj_1= np.array(pd.read_csv('./NCvsAD/adj_1.csv', header=None))[:, :].astype(float)
adj_1= torch.LongTensor(np.where(adj_1 > args.adj_Sample, 1, 0))
adj_2= np.array(pd.read_csv('./NCvsAD/adj_2.csv', header=None))[:, :].astype(float)
adj_2= torch.LongTensor(np.where(adj_2 > args.adj_Sample, 1, 0))
adj_3= np.array(pd.read_csv('./NCvsAD/adj_3.csv', header=None))[:, :].astype(float)
adj_3= torch.LongTensor(np.where(adj_3 > args.adj_Sample, 1, 0))


tr_path = './NCvsAD/X_train.csv'
tr_data = CustomDatasetWithAdj(tr_path)
te_path = './NCvsAD/X_test.csv'
te_data = CustomDatasetWithAdj(te_path)

x = te_data.data

num_epochs = 2
batch_size_ = args.batch_size_
learning_rate = 1e-3
weight_decay = 1e-4

tr_data_loader = DataLoader(dataset=tr_data, batch_size=batch_size_, shuffle=True)
te_data_loader = DataLoader(dataset=te_data, batch_size=batch_size_, shuffle=False)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loss_function = nn.CrossEntropyLoss()
input_in_dim = [116,116,116]
input_hidden_dim = [64]
network = Fusion(num_class=2, num_views=1, hidden_dim=input_hidden_dim, dropout=0.2, in_dim=input_in_dim)
# 加载模型参数
checkpoint = torch.load('./NCvsAD/test_model.pth')
network.load_state_dict(checkpoint)
network.to(device)


optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

# Initialize tracking variables
best_acc = 0.0
best_epoch = 0
best_te_f1 = 0.0
best_te_auc = 0.0
best_te_sen = 0.0
best_te_spe = 0.0

test_acc_all = []
test_auc_all = []
test_f1_all = []
test_sen_all, test_spe_all = [], []
best_model_state = None


# 评估循环
network.eval()
te_probs_all = []
te_labels_all = []
te_preds_all = []

te_feature_all = []

for i, data in enumerate(te_data_loader, 0):

    batch_x, targets = data  # 从data中获取三个邻接矩阵

    batch_x1 = batch_x[:, 0:116].reshape(-1, 116, 1).to(torch.float32).to(device)
    batch_x2 = batch_x[:, 116:232].reshape(-1, 116, 1).to(torch.float32).to(device)
    batch_x3 = batch_x[:, 232:].reshape(-1, 116, 1).to(torch.float32).to(device)
    targets = targets.long().to(device)
    adj_AHBA = adj_AHBA.to(device)
    adj_1 = adj_1.to(device)
    adj_2 = adj_2.to(device)
    adj_3 = adj_3.to(device)

    te_logits= network.infer(batch_x1, batch_x2, batch_x3,adj_AHBA,adj_1,adj_2,adj_3)
    te_prob = F.softmax(te_logits, dim=1)
    te_pre_lab = torch.argmax(te_prob, 1)

    # 统计结果
    te_probs_all.extend(te_prob[:, 1].detach().cpu().numpy())
    te_labels_all.extend(targets.detach().cpu().numpy())
    te_preds_all.extend(te_pre_lab.detach().cpu().numpy())

# 计算指标
test_acc = accuracy_score(te_labels_all, te_preds_all)
test_auc = roc_auc_score(te_labels_all, te_probs_all)
test_f1 = f1_score(te_labels_all, te_preds_all)
test_sen = sensitivity_score(te_labels_all, te_preds_all)
test_spe = specificity_score(te_labels_all, te_preds_all)


# 打印结果
print('Acc : {:.8f}'.format(test_acc))
print('F1 : {:.8f}'.format(test_f1))
print('Auc : {:.8f}'.format(test_auc))
print('Sen : {:.8f}'.format(test_sen))
print('Spe : {:.8f}'.format(test_spe))
