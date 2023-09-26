
# ind -test:
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc

# 对数据进行二进制编码：
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # dataset, labels = create_encode_dataset(filpath)
    # print(dataset.shape)
    test_dataset, test_lables = create_encode_dataset(filpath)
    print(test_dataset.shape)

    return test_dataset,test_lables




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
            dropout=0.5
        )
        # attention layer：
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2,num_heads=8,batch_first=True,dropout=0.5)

        # classfier layer：
        # self.cls_layer = nn.Linear(self.hidden_size * 2,self.num_classes)
        # self.linear=nn.Linear(output_size,self.num_classes)


        self.dropout1 = nn.Dropout(0.9)
        self.dropout2=nn.Dropout(0.3)

    def forward(self, inputs):
        input_ids = inputs
        # LSTM layer

        Bilstm_outputs, (last_hidden_state, last_cell_state) = self.Bilstm(inputs)
        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)
        # (batch_size,seq_len,hidden_size * 2)

        Bilstm_outputs = self.dropout1(Bilstm_outputs)

        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)

        context,_ = self.attention(Bilstm_outputs,Bilstm_outputs,Bilstm_outputs)
        # print("context shape:",context.shape)
        # context = self.dropout2(context)

        MutilHead_output = context

        # print("context shape:",context.shape)

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
        # 定义pooling层：
        # self.maxpool1=torch.nn.MaxPool1d(kernel_size=1,stride=2)

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
        # x=self.maxpool1(x)
        x = self.dropout1(x)

        First_outputs=x

        # 第二层卷积
        x = self.conv2(x)
        x = F.relu(x)
        # x=self.maxpool2(x)
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



import numpy as np
from numpy import interp
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    #使用各自的数据集测试每一个物种：

    batch_size = 128
    learn_rate = 0.001

    base_fpr = np.linspace(0, 1, 101)
    base_fpr[-1] = 1.0

    # 文件路径
    wheat_filepath = "../Csv/each_species_train_test/wheat_test.Csv"
    papaya_filepath = "../Csv/each_species_train_test/papaya_test.Csv"
    peanut_filepath = "../Csv/each_species_train_test/peanut_test.Csv"
    rice_filepath = '../Csv/each_species_train_test/rice_test.Csv'
    tabacum_filepath = "../Csv/each_species_train_test/tabacum_test.Csv"

    # 形成数据集：tuple ：需要修改
    test_dataset,test_labels=get_datasets(papaya_filepath)
    test_set = MyDataset(test_dataset, test_labels)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # 实例化模型和加载模型
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device='cpu'
    model = KcrNet()
    model.to(device)

    # load mmodel :需要修改
    model_path = '../model_weights/each_species_model_weights/Papaya-Weight.pth'

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    import math
    from sklearn import metrics

    test_roc = []
    test_roc_auc = []

    test_tprs = []
    test_fprs = []

    test_base_fpr = np.linspace(0, 1, 101)
    test_base_fpr[-1] = 1.0

    eval_SN_SP_ACC_MCC = []
    # 独立数据集测试：
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

            _, y_test_pred = model(x_data)

            y_test_label = torch.argmax(y_test_pred, dim=1)

            TP += ((y_true_label == 1) & (y_test_label == 1)).sum().item()
            FP += ((y_true_label == 1) & (y_test_label == 0)).sum().item()
            TN += ((y_true_label == 0) & (y_test_label == 0)).sum().item()
            FN += ((y_true_label == 0) & (y_test_label == 1)).sum().item()

            # 计算损失值：
            loss = F.cross_entropy(y_test_pred, y_data)

            # calculate acc
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_test_pred, dim=1).detach().cpu().numpy())
            # calculate auc
            auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_test_pred[:, 1].detach().cpu().numpy())

            # 计算得分
            y_test_true.append(y_data[:].detach().cpu().numpy())
            y_test_score.append(y_test_pred[:, 1].detach().cpu().numpy())

            test_acc.append(acc)
            test_loss.append(loss.detach().cpu().numpy())
            test_auc.append(auc)

        avg_acc, avg_loss, avg_auc = np.mean(test_acc), np.mean(test_loss), np.mean(test_auc)

        # 合并数据：
        y_test_true = np.concatenate(y_test_true)
        y_test_score = np.concatenate(y_test_score)

        # 保存真实值和预测值的值
        # np.save('../np_weights/PlantNh-Kcr_y_test_true.npy', y_test_true)
        # np.save('../np_weights/PlantNh-Kcr_y_test_score.npy', y_test_score)

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

        # 计算SN，SP，ACC，MCC
        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        eval_SN_SP_ACC_MCC.append(SN)
        eval_SN_SP_ACC_MCC.append(SP)
        eval_SN_SP_ACC_MCC.append(ACC)
        eval_SN_SP_ACC_MCC.append(MCC)

        # np.save('../np_weights/PlantNh-Kcr_eval.npy', eval_SN_SP_ACC_MCC)

        print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))
