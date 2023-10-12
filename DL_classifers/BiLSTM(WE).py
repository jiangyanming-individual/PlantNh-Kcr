import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import warnings
import math
from numpy import interp

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,Subset


AA_aaindex = 'ACDEFGHIKLMNPQRSTVWY'

word2id_dict={'X':0}
for i in range(len(AA_aaindex)):
    word2id_dict[AA_aaindex[i]]=i+1


train_filepath= '../Datasets/train.csv'
test_filepath= '../Datasets/ind_test.csv'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cpu'

# return squence and label
def load_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            # one_data = []
            sequence, y_label = line.strip().split(',')
            # print(line.strip().split(','))
            data.append((sequence, y_label))

    return data


train_dataset=load_data(train_filepath)
test_dataset=load_data(test_filepath)
# train_dataset.shape
# train_dataset
# print(train_dataset)


# word to id
class MyDataset(Dataset):

    def __init__(self, examples, word2id_dict):
        super(MyDataset, self).__init__()
        self.word2id_dict = word2id_dict
        self.examples = self.words_to_id(examples)

    # return squence and label
    def words_to_id(self, examples):
        temp_example = []

        for i, example in enumerate(examples):
            seq, label = example
            # use X to fill in the missing
            seq = [self.word2id_dict.get(AA, self.word2id_dict.get('X')) for AA in seq]
            # label
            label = int(label)
            temp_example.append((seq, label))

        return temp_example

    def __getitem__(self, idx):

        seq, label = self.examples[idx]

        return seq, label

    def __len__(self):
        return len(self.examples)


#train dataset：
train_set=MyDataset(train_dataset,word2id_dict)
#ind_test：
test_set=MyDataset(test_dataset,word2id_dict)

# print(train_set)



# total model
class Model_LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes=2, num_layers=1):
        super(Model_LSTM, self).__init__()
        # dict size：
        self.vocab_size = vocab_size

        # embedding_size
        self.embedding_size = embedding_size

        # hidden_size：
        self.hidden_size = hidden_size
        # num_classes：
        self.num_classes = num_classes

        # LSTM layers：
        self.num_layers = num_layers

        #embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_size
        )

        # BiLSTM：
        self.Bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # classfier layer：
        self.cls_layer = nn.Linear(self.hidden_size * 2,self.num_classes)
        self.dropout1 = nn.Dropout(0.9)


    def forward(self, inputs):
        input_ids = inputs

        # embedding layer;
        embeded_input = self.embedding(input_ids) #(batch_size,seq_len)
        print("embeded_input shape:",embeded_input.shape)
        embeded_output = embeded_input
        # LSTM layer

        Bilstm_outputs, (last_hidden_state, last_cell_state) = self.Bilstm(embeded_output)
        # print("Bilstm_outputs shape:",Bilstm_outputs.shape)

        # (batch_size,seq_len,hidden_size * 2)
        # print(Bilstm_outputs.shape)
        Bilstm_outputs = self.dropout1(Bilstm_outputs)

        # classfier layer：
        outputs = self.cls_layer(Bilstm_outputs[:,-1,:])  #[batch_size,hidden_size]

        return (embeded_output, Bilstm_outputs), outputs


# training model
warnings.filterwarnings("ignore")
# epoch
num_epochs = 30
# learning rate
learning_rate = 0.001
# the length of embedding
vocab_size = len(word2id_dict)

# embedding size
embedding_size = 10

# the hidden size of LSTM network
hidden_size = 64
num_classes = 2

num_layers = 2

train_losses = []
train_acces = []
train_auces = []


roc = []
roc_auc = []

tprs = []
fprs = []

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.0

total_SN = []
total_SP = []
total_ACC = []
total_MCC = []


def train(model, train_loader,valid_loader,device):
    print("train is start...")

    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        for batch_id, data in enumerate(train_loader):

            # print(data[0])

            data[0] = torch.stack(data[0], dim=1)
            # print("data[0]:",data[0])
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data=torch.tensor(y_data,dtype=torch.long)


            # print("data[0][0]",data[0][0])
            # print("data[0][0]",data[0][1])
            # print("label:",data[1])

            _, y_predict = model(x_data)

            # print("y_predict:",y_predict)
            # print("y_predict shape:",y_predict.shape)
            loss = F.cross_entropy(y_predict, y_data)

            # loss = loss_fn(y_predict, y_data)
            # print("loss:",loss)
            acc=metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_predict,dim=1).detach().cpu().numpy())
            # print("acc:",acc)

            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            # print("auc:",auc)
            epoch_acc.append(acc)
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_auc.append(auc)

            # print("epoch_acc:",epoch_acc)
            # print("epoch_loss:",epoch_loss)
            # print("epoch_auc:",epoch_auc)

            if (batch_id % 64 == 0):
                print("epoch is {},batch_id is {},acc is :{} ,loss is {},auc is {}".format(epoch, batch_id,acc,loss.detach().cpu().numpy(), auc))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_acc, avg_loss, avg_auc = np.mean(epoch_acc), np.mean(epoch_loss), np.mean(epoch_auc)
        print("[train:avg_acc is: {},avg_loss is: {},avg_auc is: {}]".format(avg_acc, avg_loss, avg_auc))

    # 交叉验证模型只验证一次
    print("eval is start...")
    model.eval()
    with torch.no_grad():

        valid_acc = []
        valid_loss = []
        valid_auc = []

        y_true = []
        y_score = []

        TP, FP, TN, FN = 0, 0, 0, 0

        for batch_id, data in enumerate(valid_loader):


            data[0]=torch.stack(data[0],dim=1)

            x_data=data[0].to(device)
            y_data = data[1].to(device)

            y_data=torch.tensor(y_data,dtype=torch.long)

            y_true_label = y_data

            # y_data = torch.unsqueeze(y_data, axis=-1)
            _, y_valid_pred = model(x_data)

            # y_predic_label
            y_predict_label = torch.argmax(y_valid_pred, dim=1)
            # print("y_predict_label",y_predict_label)

            #calculate loss
            loss = F.cross_entropy(y_valid_pred, y_data)

            # loss = loss_fn(y_valid_pred, y_data)
            # acc = torch.metric.accuracy(y_valid_pred, y_data)
            acc=metrics.accuracy_score(y_data.detach().cpu().numpy(),torch.argmax(y_valid_pred,dim=1).detach().cpu().numpy())

            #cal auc
            auc = metrics.roc_auc_score(y_data[:].detach().cpu().numpy(), y_valid_pred[:, 1].detach().cpu().numpy())


            # calculate score
            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_valid_pred[:, 1].detach().cpu().numpy())

            valid_acc.append(acc)
            valid_loss.append(loss.detach().cpu().numpy())
            valid_auc.append(auc)

            TP += ((y_true_label == 1) & (y_predict_label == 1)).sum().item()
            FP += ((y_true_label == 0) & (y_predict_label == 1)).sum().item()
            TN += ((y_true_label == 0) & (y_predict_label == 0)).sum().item()
            FN += ((y_true_label == 1) & (y_predict_label == 0)).sum().item()
            # valid_auc.append(auc)

            if batch_id % 64 == 0:
                print("batch_id is {} ,acc is {} ,loss is {} ,auc is {}".format(batch_id, acc, loss.detach().cpu().numpy(),auc))

        avg_acc, avg_loss, avg_auc = np.mean(valid_acc), np.mean(valid_loss), np.mean(valid_auc)

        # concate data
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        res_auc = metrics.auc(fpr, tpr)
        roc.append([fpr, tpr])

        roc_auc.append(res_auc)

        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        fprs.append(fpr)

        print("res_auc :",res_auc)
        print("[valid:avg_acc is:{},avg_loss is :{},avg_auc is:{}]".format(avg_acc, avg_loss, avg_auc))


        np.save('../np_weights/BiLSTM(WE)_y_train.npy',y_true)
        np.save('../np_weights/BiLSTM(WE)_y_train_score.npy',y_score)
        # calculate SN，SP，ACC，MCC

        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        print("Train TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("Train SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))

        total_SN.append(SN)
        total_SP.append(SP)
        total_ACC.append(ACC)
        total_MCC.append(MCC)



# five-fold cross-validation
from sklearn.model_selection import KFold
from sklearn import metrics
from torch.utils.data import Subset

# define DataLoader
from torch.utils.data import DataLoader

kf = KFold(n_splits=5, shuffle=True)
fold = 1

for train_index, valid_index in kf.split(train_set):
    print(f"第{fold}次交叉验证")

    batch_size = 128
    #create training and valid sets

    train_dataset = Subset(train_set, train_index)
    valid_dataset = Subset(train_set, valid_index)


    # DataLoader:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # use model
    model = Model_LSTM(vocab_size, embedding_size, hidden_size, num_classes, num_layers)

    model.to(device)
    train(model, train_loader,valid_loader,device)

    #save model
    torch.save(model.state_dict(), '../DL_weights/'+str(fold) + '_BiLSTM(WE)_kfold_model.pth'.format(fold))

    fold += 1


np.save('../np_weights/BiLSTM(WE)_roc_auc.npy', roc_auc)
np.save('../np_weights/BiLSTM(WE)_roc.npy', roc)


#the results of SN, SP, ACC, MCC on five-fold cross-validation
mean_SN=np.mean(total_SN)
mean_SP=np.mean(total_SP)
mean_ACC=np.mean(total_ACC)
mean_MCC=np.mean(total_MCC)


kfold_SN_SP_ACC_MCC=[]
kfold_SN_SP_ACC_MCC.append(mean_SN)
kfold_SN_SP_ACC_MCC.append(mean_SP)
kfold_SN_SP_ACC_MCC.append(mean_ACC)
kfold_SN_SP_ACC_MCC.append(mean_MCC)

#results
np.save('../np_weights/BiLSTM(WE)_SN_SP_ACC_MCC.npy', kfold_SN_SP_ACC_MCC)
print("5kfold: SN is: {}, SP is: {}, ACC is: {},MCC is: {}".format(mean_SN,mean_SP,mean_ACC,mean_MCC))



# the visualization of five-fold cross-validation

def Kf_show(plt, base_fpr, roc, roc_auc):

    for i, item in enumerate(roc):
        fpr, tpr = item
        plt.plot(fpr, tpr, label="ROC fold {} (AUC={:.4f})".format(i + 1, roc_auc[i]), lw=1, alpha=0.3)

    # mean
    plt.plot(base_fpr, np.average(tprs, axis=0),
             label=r'Mean ROC(AUC=%0.2f $\pm$ %0.2f)' % (np.mean(roc_auc), np.std(roc_auc)),
             lw=1, alpha=0.8, color='b')
    # base line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.savefig('../figures/BiLSTM(WE).png')
    plt.show()

Kf_show(plt, base_fpr, roc, roc_auc)


############################################
# total dataset to training
warnings.filterwarnings("ignore")
# epoch
num_epochs = 30
# learning rate
learning_rate = 0.001
# the length of embedding
vocab_size = len(word2id_dict)

# embedding size
embedding_size = 10

# the hidden size of LSTM network
hidden_size = 64
num_classes = 2

num_layers = 2

train_losses = []
train_acces = []
train_auces = []

roc = []
roc_auc = []

tprs = []
fprs = []

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.0

total_SN = []
total_SP = []
total_ACC = []
total_MCC = []


def total_train(model, train_loader, device):
    print("train is start...")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):

        epoch_loss = []
        epoch_acc = []
        epoch_auc = []
        for batch_id, data in enumerate(train_loader):

            # print(data[0])

            data[0] = torch.stack(data[0], dim=1)
            # print("data[0]:",data[0])
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)

            # print("data[0][0]",data[0][0])
            # print("data[0][0]",data[0][1])
            # print("label:",data[1])

            _, y_predict = model(x_data)

            # print("y_predict:",y_predict)
            # print("y_predict shape:",y_predict.shape)
            loss = F.cross_entropy(y_predict, y_data)

            # loss = loss_fn(y_predict, y_data)
            # print("loss:",loss)
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         torch.argmax(y_predict, dim=1).detach().cpu().numpy())
            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            # print("auc:",auc)
            epoch_acc.append(acc)
            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_auc.append(auc)

            # print("epoch_acc:",epoch_acc)
            # print("epoch_loss:",epoch_loss)
            # print("epoch_auc:",epoch_auc)

            if (batch_id % 64 == 0):
                print("epoch is {},batch_id is {},acc is :{} ,loss is {},auc is {}".format(epoch, batch_id, acc,
                                                                                           loss.detach().cpu().numpy(),
                                                                                           auc))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_acc, avg_loss, avg_auc = np.mean(epoch_acc), np.mean(epoch_loss), np.mean(epoch_auc)
        print("[train:avg_acc is: {},avg_loss is: {},avg_auc is: {}]".format(avg_acc, avg_loss, avg_auc))

        if (epoch + 1) == num_epochs:
            #save model
            torch.save(model.state_dict(), '../DL_weights/BiLSTM(WE)-FinalWeight.pth')


from sklearn import metrics
# DataLoader
from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model_LSTM(vocab_size, embedding_size, hidden_size, num_classes, num_layers)
model.to(device)

# training model
total_train(model, train_loader, device)




