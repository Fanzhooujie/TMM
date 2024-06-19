from model import *
from sklearn.metrics import roc_auc_score, f1_score
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

num_epochs = 2000
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

train_loss_all, train_acc_all, test_acc_all = [], [], []
train_auc_all, test_auc_all = [], []
train_f1_all, test_f1_all = [], []
test_sen_all, test_spe_all = [], []


for epoch in range(0, num_epochs):
    # Print epoch
    print(' Epoch {}/{}'.format(epoch, num_epochs - 1))
    print("-" * 10)
    # Set current loss value
    network.train()
    current_loss = 0.0
    train_loss = 0.0
    train_corrects = 0
    train_num = 0
    tr_probs_all = []
    tr_labels_all = []
    tr_preds_all = []

    for i, data in enumerate(tr_data_loader, 0):

        batch_x, targets = data  # 从data中获取三个邻接矩阵

        batch_x1 = batch_x[:, 0:116].reshape(-1, 116, 1).to(torch.float32).to(device)
        batch_x2 = batch_x[:, 116:232].reshape(-1, 116, 1).to(torch.float32).to(device)
        batch_x3 = batch_x[:, 232:].reshape(-1, 116, 1).to(torch.float32).to(device)
        targets = targets.long().to(device)
        adj_AHBA = adj_AHBA.to(device)
        adj_1 = adj_1.to(device)
        adj_2 = adj_2.to(device)
        adj_3 = adj_3.to(device)
        optimizer.zero_grad()
        loss_fusion, tr_logits, gat_output1, gat_output2, gat_output3, output1, output2, output3 = network(
            batch_x1, batch_x2, batch_x3,adj_AHBA,adj_1,adj_2,adj_3,targets)

        tr_prob = F.softmax(tr_logits, dim=1)
        tr_pre_lab = torch.argmax(tr_prob, 1)

        # 计算损失并进行反向传播
        loss = loss_fusion
        loss.backward()
        optimizer.step()

        # 计算训练损失和准确率
        train_loss += loss.item() * batch_x1.size(0)
        train_corrects += torch.sum(tr_pre_lab == targets.data)
        train_num += batch_x1.size(0)

        # 记录训练结果
        with torch.no_grad():
            # 推断模型并计算预测概率和预测类别
            tr_logits = network.infer(batch_x1, batch_x2, batch_x3,adj_AHBA,adj_1,adj_2,adj_3)
            tr_prob = F.softmax(tr_logits, dim=1)
            tr_preds = torch.argmax(tr_prob, 1)

        # 记录预测概率、真实标签和预测标签
        tr_probs_all.extend(tr_prob[:, 1].cpu().numpy())  # 选择正类别的概率
        tr_labels_all.extend(targets.cpu().numpy())
        tr_preds_all.extend(tr_pre_lab.cpu().numpy())


    # Evaluationfor this fold
    network.eval()
    test_loss = 0.0
    test_corrects = 0
    test_num = 0
    te_probs_all = []
    te_labels_all = []
    te_preds_all = []

    for i, data in enumerate(te_data_loader, 0):
        batch_x, targets = data

        batch_x1 = batch_x[:, 0:116].reshape(-1, 116, 1).to(torch.float32).to(device)
        batch_x2 = batch_x[:, 116:232].reshape(-1, 116, 1).to(torch.float32).to(device)
        batch_x3 = batch_x[:, 232:].reshape(-1, 116, 1).to(torch.float32).to(device)
        targets = targets.long().to(device)
        adj_AHBA = adj_AHBA.to(device)
        adj_1 = adj_1.to(device)
        adj_2 = adj_2.to(device)
        adj_3 = adj_3.to(device)

        te_logits = network.infer(batch_x1, batch_x2, batch_x3,adj_AHBA,adj_1,adj_2,adj_3)
        te_prob = F.softmax(te_logits, dim=1)
        te_pre_lab = torch.argmax(te_prob, 1)

        test_corrects += torch.sum(te_pre_lab == targets.data)
        test_num += batch_x1.size(0)

        with torch.no_grad():
            te_logits = network.infer(batch_x1, batch_x2, batch_x3,adj_AHBA,adj_1,adj_2,adj_3)
            te_prob = F.softmax(te_logits, dim=1)
            te_preds = torch.argmax(te_prob, 1)

        te_probs_all.extend(te_prob[:, 1].cpu().numpy())
        te_labels_all.extend(targets.cpu().numpy())
        te_preds_all.extend(te_pre_lab.cpu().numpy())
        

    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(train_corrects.double().item() / train_num)
    test_acc_all.append(test_corrects.double().item() / test_num)
    tr_auc = roc_auc_score(tr_labels_all, tr_probs_all)
    train_auc_all.append(tr_auc)
    te_auc = roc_auc_score(te_labels_all, te_probs_all)
    test_auc_all.append(te_auc)
    tr_f1 = f1_score(tr_labels_all, tr_preds_all)
    train_f1_all.append(tr_f1)
    te_f1 = f1_score(te_labels_all, te_preds_all)
    test_f1_all.append(te_f1)
    test_sen = sensitivity_score(te_labels_all, te_preds_all)
    test_sen_all.append(test_sen)
    test_spe = specificity_score(te_labels_all, te_preds_all)
    test_spe_all.append(test_spe)
    print('{} Train Loss : {:.8f} Train ACC : {:.8f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
    print('{}  Test ACC : {:.8f}'.format(epoch, test_acc_all[-1]))

