import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn import metrics


# 对数据进行二进制编码：
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 对数据集进行编码的操作：
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
        # 取一条氨基酸序列和对应的lable；
        code = []  # 一条序列
        # seq_index=1

        result_seq_labels.append(int(data[1]))
        for seq in data[0]:
            one_code = []  # 一条序列
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == seq:
                    flag = 1
                else:
                    flag = 0
                one_code.append(flag)

            code.extend(one_code)

        result_seq_datas.append(code)
        # print(one_seq_data)

    return np.array(result_seq_datas), np.array(result_seq_labels, dtype=np.int64)



# 数据集的文件路径：
def get_datasets(filpath):

    dataset, labels = create_encode_dataset(filpath)
    print(dataset.shape)
    # test_dataset, test_labels = create_encode_dataset(filpath)
    # print(test_dataset.shape)

    return dataset,labels


# 构建数据集：
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



# total model
class Model_LSTM_MutilHeadSelfAttention(nn.Module):

    def __init__(self,input_size, hidden_size, num_classes=2, num_layers=1, attention=None):
        super(Model_LSTM_MutilHeadSelfAttention, self).__init__()

        self.input_size = input_size

        # hidden_size：
        self.hidden_size = hidden_size
        # num_classes：
        self.num_classes = num_classes

        # LSTM layers：
        self.num_layers = num_layers



        # BiLSTM Layer：
        self.Bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        # attention layer：
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2,num_heads=8,batch_first=True,dropout=0.5)

        self.dropout1 = nn.Dropout(0.9)

    def forward(self, inputs):
        input_ids = inputs
        # LSTM layer

        Bilstm_outputs, (last_hidden_state, last_cell_state) = self.Bilstm(inputs)

        Bilstm_outputs = self.dropout1(Bilstm_outputs)
        context,_ = self.attention(Bilstm_outputs,Bilstm_outputs,Bilstm_outputs)

        MutilHead_output = context

        return (Bilstm_outputs, MutilHead_output), context

import warnings
# 模型训练：
warnings.filterwarnings("ignore")
# LSTM网络隐状态向量的维度

input_size=len(Amino_acid_sequence)
hidden_size = 64
num_classes = 2
num_layers = 1


class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2):
        super(KcrNet, self).__init__()
        # 定义卷积层：
        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        # self.maxpool2=torch.nn.MaxPool1d(kernel_size=1,stride=2)

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=29, kernel_size=5, padding=2, stride=2)

        self.BiLSTM_ATT=Model_LSTM_MutilHeadSelfAttention(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

        # 定义全连接层：
        self.flatten = torch.nn.Flatten()
        # 定义感知层；
        self.linear1 = torch.nn.Linear(in_features=29 * 136, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)


        self.dropout1=torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):

        inputs=x

        x = torch.permute(x, [0, 2, 1]) # 对数据进行重新排列

        # 第一层卷积
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        First_outputs=x

        # 第二层卷积
        x = self.conv2(x)
        x = F.relu(x)

        x = self.dropout2(x)
        Second_outputs=x

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        Third_outputs=x
        # print("final x shape:",x.shape)

        visual_outputs,BiLSTM_outputs=self.BiLSTM_ATT(inputs)


        total_outputs=torch.cat([x,BiLSTM_outputs],dim=-1)
        # print("total_outputs shape:",total_outputs.shape)

        # 全连接层：
        x = self.flatten(total_outputs)

        x = self.linear1(x)
        x = F.relu(x)  # 激活函数
        Linear_output=x

        x = self.linear2(x)
        return (inputs,First_outputs,Second_outputs,Third_outputs,visual_outputs,total_outputs,Linear_output),x


model=KcrNet()
print(model)


import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings

warnings.filterwarnings("ignore")

# 模型准备
epochs = 50
batch_size = 128
learn_rate = 0.001

# 使用低级api进行构建训练；
train_loss = []
train_acc = []
train_auc = []

eval_losses = []
eval_accuracies = []
eval_auces = []

roc = []
roc_auc = []

tprs = []
fprs = []

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.0


def total_train(model, train_loader, device):
    print("train is start!")
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
    # loss_fn = FocalLoss(alpha=0.8, gamma=1.5)
    for epoch in range(epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        for batch_id, data in enumerate(train_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            # print(x_data)

            # 进行模型训练：
            _, y_predict = model(x_data)
            # print("y_predict:",y_predict)

            # 二分问题使用的损失函数 binary_cross_entropy函数；
            loss = F.cross_entropy(y_predict, y_data)

            # loss=loss_fn(y_predict,y_data)

            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_predict, dim=1).detach().cpu().numpy())
            # print("y_data:",y_data)
            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_acc.append(acc)
            epoch_auc.append(auc)
            if (batch_id % 64 == 0):
                print("epoch is :{},batch_id is {},loss is {},acc is:{},auc is:{}".format(epoch, batch_id,
                                                                                          loss.detach().cpu().numpy(),
                                                                                          acc, auc))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 取每次epoch的均值
        avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
        print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))
        #
        if (epoch + 1) == epochs:
            # 保存模型：需要修改
            # torch.save(model.state_dict(), '../model_weights/each_species_model_weights/Wheat-Weight.pth')
            # torch.save(model.state_dict(), '../model_weights/each_species_model_weights/Rice-Weight.pth')
            # torch.save(model.state_dict(), '../model_weights/each_species_model_weights/Tabacum-Weight.pth')
            # torch.save(model.state_dict(), '../model_weights/each_species_model_weights/Peanut-Weight.pth')
            torch.save(model.state_dict(), '../model_weights/each_species_model_weights/Papaya-Weight.pth')


if __name__ == '__main__':

    #filepath
    wheat_train_filepath= '../Csv/each_species_train_test/wheat_train.Csv'
    rice_train_filepath= '../Csv/each_species_train_test/rice_train.Csv'
    tabacum_train_filepath= '../Csv/each_species_train_test/tabacum_train.Csv'
    peanut_train_filepath= '../Csv/each_species_train_test/peanut_train.Csv'
    papaya_train_filepath= '../Csv/each_species_train_test/papaya_train.Csv'


    # 形成数据集：tuple ：需要修改
    train_dataset,train_labels=get_datasets(papaya_train_filepath)
    train_set = MyDataset(train_dataset, train_labels)


    # 形成DataLoader:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device='cpu'
    model = KcrNet()
    model.to(device)
    # 训练model
    total_train(model, train_loader, device)
