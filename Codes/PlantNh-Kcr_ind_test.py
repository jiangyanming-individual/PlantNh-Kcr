# author:Lenovo
# datetime:2023/7/22 20:28
# software: PyCharm
# project:PlantNh-Kcr

"""
independent test
"""
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import interp
import warnings
warnings.filterwarnings("ignore")

#amino acid sequence
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'

#binary encoding
def create_encode_dataset(filepath):
    data_list = []
    result_seq_datas = []
    result_seq_labels = []
    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            x_data_sequence, label = list(line.strip('\n').split(','))
            data_list.append((x_data_sequence, label))
    for data in data_list:
        code = []  # one sequence
        result_seq_labels.append(int(data[1]))
        for seq in data[0]:
            one_code = []
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == seq:
                    flag = 1
                else:
                    flag = 0
                one_code.append(flag)
            code.extend(one_code)
        result_seq_datas.append(code)
    return np.array(result_seq_datas), np.array(result_seq_labels, dtype=np.int32)


train_filepath='../Datasets/train.csv'
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

    def __init__(self,input_size, hidden_size, num_classes=2, num_layers=1):
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
        # print("context shape:",context.shape)
        return (Bilstm_outputs, MutilHead_output), context


class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2,input_size=21,hidden_size=64,num_layers=1):
        super(KcrNet, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=29, kernel_size=5, padding=2, stride=2)
        self.BiLSTM_ATT=Model_LSTM_MutilHeadSelfAttention(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)

        # flatten layer
        self.Flatten = torch.nn.Flatten()
        # lienar layer
        self.Linear1 = torch.nn.Linear(in_features=29 * 136, out_features=128)
        self.Linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)

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

        #the outpur of BiLSTM and attention layers
        visual_outputs,BiLSTM_outputs=self.BiLSTM_ATT(inputs)
        #concate:
        total_outputs=torch.cat([x,BiLSTM_outputs],dim=-1)
        #flatten layer
        x = self.Flatten(total_outputs)
        x = self.Linear1(x)
        x = F.relu(x)  #activate funcation
        Linear_output=x
        x = self.Linear2(x)

        return (inputs,First_outputs,Second_outputs,Third_outputs,total_outputs,Linear_output),x


def Calculate_confusion_matrix(y_test_true,y_pred_label):

    conf_matrix = confusion_matrix(y_test_true, y_pred_label)
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1Score = (2 * TP) / float(2 * TP + FP + FN)

    return (TN,TP,FN,FP),(SN,SP,ACC,MCC,F1Score)

# mean and std metrics values:
def Calcuate_mean_std_metrics_values(total_SN,total_SP,total_ACC,total_F1_score,total_MCC,total_AUC):
    # Calculate mean and std metrics values:
    mean_SN = np.mean(total_SN)
    mean_SP = np.mean(total_SP)
    mean_ACC = np.mean(total_ACC)
    mean_F1_score = np.mean(total_F1_score)
    mean_MCC = np.mean(total_MCC)
    mean_AUC = np.mean(total_AUC)

    std_SN = np.std(total_SN)
    std_SP = np.std(total_SP)
    std_ACC = np.std(total_ACC)
    std_F1_score = np.std(total_F1_score)
    std_MCC = np.std(total_MCC)
    std_AUC = np.std(total_AUC)

    mean_metrics = []
    mean_metrics.append(mean_SN)
    mean_metrics.append(mean_SP)
    mean_metrics.append(mean_ACC)
    mean_metrics.append(mean_F1_score)
    mean_metrics.append(mean_MCC)
    mean_metrics.append(mean_AUC)

    std_metrics = []
    std_metrics.append(std_SN)
    std_metrics.append(std_SP)
    std_metrics.append(std_ACC)
    std_metrics.append(std_F1_score)
    std_metrics.append(std_MCC)
    std_metrics.append(std_AUC)

    np.save('../np_weights/PlantNh-Kcr_ind_test_mean_Metric.npy', mean_metrics)
    np.save('../np_weights/PlantNh-Kcr_ind_test_std_Metric.npy', std_metrics)
    print(
        "ind test Mean metrics : SN is {:.3f},SP is {:.3f},ACC is {:.3f},F1-score is {:.3f},MCC is {:.3f},AUC is {:.3f}".
        format(mean_SN, mean_SP, mean_ACC, mean_F1_score, mean_MCC, mean_AUC))
    print(
        "ind test std metrics : SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".
        format(std_SN, std_SP, std_ACC, std_F1_score, std_MCC, std_AUC))

def total_train(model, train_loader, optimizer,train_criterion,device,fold):

    print("train is start!")
    model.train()
    for epoch in range(epochs):
        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        for batch_id, data in enumerate(train_loader):
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            _, y_predict = model(x_data)
            loss = train_criterion(y_predict,y_data)
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_predict, dim=1).detach().cpu().numpy())
            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_acc.append(acc)
            epoch_auc.append(auc)
            if (batch_id % 64 == 0):
                print("epoch is :{},batch_id is {},loss is {},acc is:{},auc is:{}".format(epoch+1, batch_id,
                                                                                          loss.detach().cpu().numpy(),
                                                                                          acc, auc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss, avg_acc, avg_auc = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_auc)
        print("[train acc is:{}, loss is :{},auc is:{}]".format(avg_acc, avg_loss, avg_auc))
        if (epoch + 1) == epochs:
            # save model
            torch.save(model.state_dict(),'../model_weights/ind_test/' + str(fold) + '_PlantNh-Kcr-FinalWeight.pth')

def independet_test(model,test_loader,criterion,device):

    model.eval()
    with torch.no_grad():
        test_acc = []
        test_loss = []
        test_auc = []

        y_test_true = []
        y_test_score = []
        y_predict_labels_list=[]
        for batch_id, data in enumerate(test_loader):
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            _,y_test_pred = model(x_data)
            y_predict_label = torch.argmax(y_test_pred, dim=1)
            y_predict_labels_list.append(y_predict_label.detach().cpu().numpy())
            #calculate loss
            loss = criterion(y_test_pred,y_data)
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
        y_test_true_list = np.concatenate(y_test_true)
        y_test_score_list = np.concatenate(y_test_score)
        y_test_pred_label = np.concatenate(y_predict_labels_list)

        # save the score values and true values
        np.save('../np_weights/PlantNh-Kcr_y_test_true.npy', y_test_true_list)
        np.save('../np_weights/PlantNh-Kcr_y_test_score.npy', y_test_score_list)

        fpr, tpr, _ = roc_curve(y_test_true_list, y_test_score_list)
        res_auc = metrics.auc(fpr, tpr)
        test_roc.append([fpr, tpr])
        test_roc_auc_areas.append(res_auc)

        #mean tpr
        tpr = interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

        print("[ind_test:avg_acc is:{},avg_loss is :{},AUC is:{}]".format(avg_acc, avg_loss, res_auc))
        #confusion matrix
        (TN,TP,FN,FP),(SN,SP,ACC,MCC,F1Score)=Calculate_confusion_matrix(y_test_true_list,y_test_pred_label)

        total_SN.append(SN)
        total_SP.append(SP)
        total_ACC.append(ACC)
        total_F1_score.append(F1Score)
        total_MCC.append(MCC)
        total_AUC.append(res_auc)

        print('--------------------------------------independent test---------------------------------------')
        print("[ind_test TP is {},FP is {},TN is {},FN is {}]".format(TP, FP, TN, FN))
        print(
            "ind_test: SN is {},SP is {},ACC is {},F1-score is {},MCC is {},AUC is {}".
            format(SN, SP, ACC, F1Score, MCC,res_auc))
        print('---------------------------------------------------------------------------------------------')

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25,gamma=2,reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction=reduction
    def forward(self, input, target):
        # input:size is N * 2. N　is the batch　size,
        # target:size is N. N is the batch size

        #claculate passibility:
        eps=1e-7
        pt = torch.softmax(input, dim=1)
        #positive passibility:
        p = pt[:, 1]
        loss = -self.alpha* torch.pow((1-p),self.gamma) * (target * torch.log(p + eps)) - \
               (1 - self.alpha) * torch.pow(p,self.gamma) * ((1 - target) * torch.log(1 - p + eps))
        if self.reduction == 'sum':
            loss=loss.sum() # sum
        else:
            loss=loss.mean() # mean
        return loss

def random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def Kf_AUROC_show(plt, mean_fpr, tprs,roc_auc_areas):

    plt.figure(dpi=1000)
    #calculate mean value
    mean_tpr=np.mean(tprs,axis=0)
    np.save('../np_weights/PlantNh-Kcr_test_mean_fprs.npy', mean_fpr)
    np.save('../np_weights/PlantNh-Kcr_test_mean_tprs.npy', mean_tpr)
    np.save('../np_weights/PlantNh-Kcr_test_AUCs.npy', roc_auc_areas)

    plt.plot(mean_fpr, mean_tpr,
             label=r'PlantNh-Kcr (AUC={:.1%} $\pm$ {:.2%})'.format(np.mean(roc_auc_areas), np.std(roc_auc_areas)),
             lw=1, alpha=0.8, color='r')
    #base line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.savefig('../figures/PlantNh-Kcr-ind_test.jpg')
    plt.show()


if __name__ == '__main__':

    # total_metrics
    total_SN = []
    total_SP = []
    total_ACC = []
    total_F1_score = []
    total_MCC = []
    total_AUC = []

    # figure mean AUC:
    test_roc = []
    test_roc_auc_areas = []
    mean_fpr = np.linspace(0, 1, 101)
    mean_fpr[-1] = 1.0
    tprs = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    learn_rate = 0.001
    epochs = 50

    input_size = len(Amino_acid_sequence)
    hidden_size = 64
    num_classes = 2
    num_layers = 1

    train_set = MyDataset(train_dataset, train_labels)
    test_set = MyDataset(test_dataset, test_labels)

    train_criterion =FocalLoss(alpha=0.7,gamma=1)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

    seed=800
    for i in range(10):
        print("第{}次独立测试".format(i+1))
        random_seeds(seed)
        seed+=10
        model = KcrNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
        model.to(device)
        print(model)
        # training model
        total_train(model, train_loader,optimizer,train_criterion,device,i+1)
        # test model
        model_path = '../model_weights/ind_test/' + str(i+1) + '_PlantNh-Kcr-FinalWeight.pth'
        model.load_state_dict(torch.load(model_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        independet_test(model,test_loader,criterion,device)
    # mean, std metrics values：
    Calcuate_mean_std_metrics_values(total_SN, total_SP, total_ACC, total_F1_score, total_MCC, total_AUC)
    #ROC_AUC curve
    Kf_AUROC_show(plt, mean_fpr,tprs,test_roc_auc_areas)

