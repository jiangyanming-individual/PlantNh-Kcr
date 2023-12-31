# author:Lenovo
# datetime:2023/7/22 20:28
# software: PyCharm
# project:PlantNh-Kcr

"""
independent test
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
import numpy as np
import math
from numpy import interp




#amino acid sequence
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'


#binary encoding
def create_encode_dataset(filepath):
    data_list = []

    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            sequence, label = list(line.strip('\n').split(','))
            data_list.append((sequence, label))

    result_seq_data = []
    result_seq_labels = []

    for data in data_list:
        seq,label=data[0],data[1]
        one_seq = []
        result_seq_labels.append(int(label))
        for amino_acid in seq:
            one_amino_acid = []
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == amino_acid:
                    flag = 1
                else:
                    flag = 0
                one_amino_acid.append(flag)
            one_seq.extend(one_amino_acid)
        result_seq_data.append(one_seq)
    return np.array(result_seq_data), np.array(result_seq_labels, dtype=np.int64)


train_filepath='../Datasets/train.csv'
#test file path
test_filepath= '../Datasets/ind_test.csv'


train_dataset, train_labels = create_encode_dataset(train_filepath)
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
        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size * 2, self.num_classes)
        self.dropout1 = nn.Dropout(0.9)

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


input_size=len(Amino_acid_sequence)
hidden_size = 64
num_classes = 2
num_layers = 1


class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2):
        super(KcrNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)


        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)

        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=29, kernel_size=5, padding=2, stride=2)

        self.BiLSTM_ATT=Model_LSTM_MutilHeadSelfAttention(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

        # flatten layer
        self.flatten = torch.nn.Flatten()
        # lienar layer
        self.linear1 = torch.nn.Linear(in_features=29 * 136, out_features=128)
        self.linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)


        self.dropout1=torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):

        inputs=x

        x = torch.permute(x, [0, 2, 1]) #permute

        # first conv1d
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        First_outputs=x

        #seconde conv1d
        x = self.conv2(x)
        x = F.relu(x)

        x = self.dropout2(x)
        Second_outputs=x

        #third conv1d
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        Third_outputs=x
        # print("final x shape:",x.shape)

        #the outpur of BiLSTM and attention layers
        visual_outputs,BiLSTM_outputs=self.BiLSTM_ATT(inputs)


        #concate:
        total_outputs=torch.cat([x,BiLSTM_outputs],dim=-1)
        # print("total_outputs shape:",total_outputs.shape)

        #flatten layer
        x = self.flatten(total_outputs)

        x = self.linear1(x)
        x = F.relu(x)  #activate funcation
        Linear_output=x

        x = self.linear2(x)

        #the output of each layer and the final output
        return (inputs,First_outputs,Second_outputs,Third_outputs,visual_outputs,Linear_output),x


model=KcrNet()
print(model)


def total_train(model, train_loader, device):
    print("train is start!")
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
    for epoch in range(epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        for batch_id, data in enumerate(train_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            # print(x_data)
            _, y_predict = model(x_data)

            loss = F.cross_entropy(y_predict, y_data)

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

        avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
        print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))

        if (epoch + 1) == epochs:
            # save model
            torch.save(model.state_dict(), '../model_weights/PlantNh-Kcr-FinalWeight.pth')


def independet_test(model,test_loader,device):

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

            #calculate loss
            loss = F.cross_entropy(y_test_pred, y_data)

            #calculate acc
            acc =metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_test_pred,dim=1).detach().cpu().numpy())
            #calculate auc
            auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_test_pred[:, 1].detach().cpu().numpy())


            #calculate sscore
            y_test_true.append(y_data[:].detach().cpu().numpy())
            y_test_score.append(y_test_pred[:, 1].detach().cpu().numpy())

            test_acc.append(acc)
            test_loss.append(loss.detach().cpu().numpy())
            test_auc.append(auc)

        avg_acc, avg_loss, avg_auc = np.mean(test_acc), np.mean(test_loss), np.mean(test_auc)

        # concate data
        y_test_true = np.concatenate(y_test_true)
        y_test_score = np.concatenate(y_test_score)

        # save the score values and true values
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

        # calculate SN，SP，ACC，MCC
        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        eval_SN_SP_ACC_MCC.append(SN)
        eval_SN_SP_ACC_MCC.append(SP)
        eval_SN_SP_ACC_MCC.append(ACC)
        eval_SN_SP_ACC_MCC.append(MCC)

        np.save('../np_weights/PlantNh-Kcr_eval.npy', eval_SN_SP_ACC_MCC)
        print("ind_test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("ind_test : SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))



if __name__ == '__main__':

    batch_size = 128
    learn_rate = 0.001
    epochs = 50

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    train_set = MyDataset(train_dataset, train_labels)
    test_set = MyDataset(test_dataset, test_labels)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
    model = KcrNet()
    model.to(device)


    # training model
    total_train(model, train_loader, device)

    #-------------------------------------------------------------------------->
    # test different species:

    # different plants
    wheat_filepath = "../Csv/speices_train_test_sets/test/wheat_test.csv"
    papaya_filepath = "../Csv/speices_train_test_sets/test/papaya_test.csv"
    peanut_filepath = "../Csv/speices_train_test_sets/test/peanut_test.csv"
    rice_filepath = '../Csv/speices_train_test_sets/test/rice_test.csv'
    tabacum_filepath = "../Csv/speices_train_test_sets/test/tabacum_test.csv"

    # wheat:
    # wheat_test_dataset, wheat_test_labels = create_encode_dataset(wheat_filepath)
    # print(wheat_test_dataset.shape)
    # wheat_test_set = MyDataset(wheat_test_dataset, wheat_test_labels)
    # test_loader = DataLoader(wheat_test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # tabacum:
    # tabacum_test_dataset, tabacum_test_labels = create_encode_dataset(tabacum_filepath)
    # print(tabacum_test_dataset.shape)
    # tabacum_test_set = MyDataset(tabacum_test_dataset, tabacum_test_labels)
    # test_loader = DataLoader(tabacum_test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # rice:
    # rice_test_dataset, rice_test_labels = create_encode_dataset(rice_filepath)
    # print(rice_test_dataset.shape)
    # rice_test_set = MyDataset(rice_test_dataset, rice_test_labels)
    # test_loader = DataLoader(rice_test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # peanut:

    # peanut_test_dataset, peanut_test_labels = create_encode_dataset(peanut_filepath)
    # print(peanut_test_dataset.shape)
    # peanut_test_set = MyDataset(peanut_test_dataset, peanut_test_labels)
    # test_loader = DataLoader(peanut_test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    # papaya:
    # papaya_test_dataset, papaya_test_labels = create_encode_dataset(papaya_filepath)
    # print(papaya_test_dataset.shape)
    # papaya_test_set = MyDataset(papaya_test_dataset, papaya_test_labels)
    # test_loader = DataLoader(papaya_test_set, batch_size=batch_size, shuffle=True, drop_last=False)
    #----------------------------------------------------------------------------------->


    # to independent test model
    #load model
    model_path= '../model_weights/PlantNh-Kcr-FinalWeight.pth'
    model.load_state_dict(torch.load(model_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    independet_test(model,test_loader,device)

