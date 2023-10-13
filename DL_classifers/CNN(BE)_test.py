# author:Lenovo
# datetime:2023/7/22 20:28
# software: PyCharm


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc


Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cpu'


def create_encode_dataset(filepath):
    data_list = []
    result_seq_datas = []
    result_seq_labels = []
    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            x_data_sequence, label = list(line.strip('\n').split(','))
            data_list.append((x_data_sequence, label))

        # print(data_list)
        # print(len(data_list))

    for data in data_list:

        code = []

        result_seq_labels.append(int(data[1]))
        for seq in data[0]:
            one_code = []  #
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == seq:
                    flag = 1
                else:
                    flag = 0
                one_code.append(flag)

            code.extend(one_code)

        # one_seq_data=np.array(code).astype('float32').reshape((1,29,29))
        result_seq_datas.append(code)
        # print(one_seq_data)

    return np.array(result_seq_datas), np.array(result_seq_labels, dtype=np.int64)


train_filepath = '../Datasets/train.csv'
test_filepath = '../Datasets/ind_test.csv'


import numpy as np
train_dataset, train_labels = create_encode_dataset(train_filepath)
print(train_dataset.shape)
test_dataset, test_labels = create_encode_dataset(test_filepath)
print(test_dataset.shape)


class MyDataset(Dataset):

    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        x_data = np.array(self.datas[index]).astype('float32').reshape((29, 21))
        y_data = self.labels[index]

        return x_data, y_data

    def __len__(self):
        return len(self.datas)


train_set = MyDataset(train_dataset, train_labels)
test_set = MyDataset(test_dataset, test_labels)


class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2):
        super(KcrNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        # self.maxpool2=torch.nn.MaxPool1d(kernel_size=1,stride=2)

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        # self.maxpool3=torch.nn.MaxPool1d(kernel_size=3,stride=1)
        # self.attention=QKV_SelfAttention()
        self.flatten = torch.nn.Flatten()

        self.linear1 = torch.nn.Linear(in_features=32 * 8, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)

        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):

        inputs = x
        x = torch.permute(x, [0, 2, 1])

        x = self.conv1(x)
        x = F.relu(x)
        # x=self.maxpool1(x)
        x = self.dropout1(x)
        First_outputs = x

        x = self.conv2(x)
        x = F.relu(x)
        # x=self.maxpool2(x)
        x = self.dropout1(x)
        Second_outputs = x

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        Third_outputs = x

        x = self.flatten(x)

        x = self.linear1(x)
        x = F.relu(x)
        Linear_output = x
        x = self.linear2(x)
        return (inputs,First_outputs, Second_outputs, Third_outputs,Linear_output), x

model=KcrNet()
print(model)


import numpy as np
from numpy import interp
import warnings
warnings.filterwarnings("ignore")

epochs = 30
batch_size = 128
learn_rate = 0.001


base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.0

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

model = KcrNet()
model.to(device)


#load mmodel
model_path= "../DL_weights/CNN(BE)-FinalWeight.pth"
model.load_state_dict(torch.load(model_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))




import math
from sklearn import metrics

test_roc = []
test_roc_auc = []

test_tprs = []
test_fprs = []

test_base_fpr = np.linspace(0, 1, 101)
test_base_fpr[-1] = 1.0


eval_SN_SP_ACC_MCC=[]
model.eval()
with torch.no_grad():
    test_acc = []
    test_loss = []
    test_auc = []

    y_test_true = []
    y_test_score = []

    TP, FP, TN, FN = 0, 0, 0, 0

    SN = 0
    SP = 0
    ACC = 0
    MCC = 0

    for batch_id, data in enumerate(test_loader):
        x_data = data[0].to(device)
        y_data = data[1].to(device)

        y_data = torch.tensor(y_data, dtype=torch.long)

        y_true_label = y_data
        # y_data = torch.unsqueeze(y_data, axis=-1)

        _,y_test_pred = model(x_data)

        y_test_label = torch.argmax(y_test_pred, dim=1)

        TP += ((y_true_label == 1) & (y_test_label == 1)).sum().item()
        FP += ((y_true_label == 0) & (y_test_label == 1)).sum().item()
        TN += ((y_true_label == 0) & (y_test_label == 0)).sum().item()
        FN += ((y_true_label == 1) & (y_test_label == 0)).sum().item()

        loss = F.cross_entropy(y_test_pred, y_data)

        #calculate acc
        acc =metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_test_pred,dim=1).detach().cpu().numpy())
        #calculate auc
        auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_test_pred[:, 1].detach().cpu().numpy())

        y_test_true.append(y_data[:].detach().cpu().numpy())
        y_test_score.append(y_test_pred[:, 1].detach().cpu().numpy())

        test_acc.append(acc)
        test_loss.append(loss.detach().cpu().numpy())
        test_auc.append(auc)

    avg_acc, avg_loss, avg_auc = np.mean(test_acc), np.mean(test_loss), np.mean(test_auc)

    y_test_true = np.concatenate(y_test_true)
    y_test_score = np.concatenate(y_test_score)

    np.save('../np_weights/CNN(BE)_y_test_true.npy', y_test_true)
    np.save('../np_weights/CNN(BE)_y_test_score.npy', y_test_score)

    fpr, tpr, _ = metrics.roc_curve(y_test_true, y_test_score)

    auc = metrics.auc(fpr, tpr)

    test_roc.append([fpr, tpr])
    test_roc_auc.append(auc)

    tpr = interp(test_base_fpr, fpr, tpr)
    tpr[0] = 0.0
    test_tprs.append(tpr)
    test_fprs.append(fpr)

    print("[ind_test:avg_acc is:{},avg_loss is :{},auc is:{}]".format(avg_acc, avg_loss, auc))

    # eval_SN_SP_ACC_MCC=[]

    #SN，SP，ACC，MCC
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    eval_SN_SP_ACC_MCC.append(SN)
    eval_SN_SP_ACC_MCC.append(SP)
    eval_SN_SP_ACC_MCC.append(ACC)
    eval_SN_SP_ACC_MCC.append(MCC)

    np.save('../np_weights/CNN(BE)_eval.npy', eval_SN_SP_ACC_MCC)

    print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))


