

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
#the filepath of training and test set
train_filepath= '../Datasets/train.csv'
test_filepath= '../Datasets/ind_test.csv'

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
        # get sequence and label
        code = []

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
    return np.array(result_seq_datas), np.array(result_seq_labels, dtype=np.int64)


# train_dataset, train_labels = create_encode_dataset(train_filepath)
# print(train_dataset.shape)
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


# train_set = MyDataset(train_dataset, train_labels)
test_set = MyDataset(test_dataset, test_labels)

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


input_size=len(Amino_acid_sequence)
hidden_size = 64
num_classes = 2
num_layers = 1

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

        #the output of each layer and the final output
        return (inputs,First_outputs,Second_outputs,Third_outputs,total_outputs,Linear_output),x



batch_size = 256
learn_rate = 0.001
base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = KcrNet(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
model.to(device)

#load mmodel
model_path= "../model_weights/ind_test/6_PlantNh-Kcr-FinalWeight.pth"
model.load_state_dict(torch.load(model_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# t-SNE
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

test_inputs_features = []
test_First_Conv1D_features = []
test_Third_Conv1D_features = []
test_Flatten_features = []
test_Linear_features = []

labels = []

for bacth_id, data in enumerate(test_loader):

    x_data = data[0].to(device)
    y_data = data[1].to(device)
    y_data = torch.tensor(y_data, dtype=torch.long)
    features, output = model(x_data)

    # print(len(features))
    # print(features[0])
    # print(features[4].shape)
    test_inputs_features.append(features[0].detach().cpu().numpy())
    test_First_Conv1D_features.append(features[1].detach().cpu().numpy())
    test_Third_Conv1D_features.append(features[3].detach().cpu().numpy())
    test_Flatten_features.append(features[4].detach().cpu().numpy())
    test_Linear_features.append(features[5].detach().cpu().numpy())
    labels.append(y_data.detach().cpu().numpy())  # labels

test_inputs_features = np.concatenate(test_inputs_features, axis=0)
print("test_inputs_features.shape:", test_inputs_features.shape)

test_First_Conv1D_features = np.concatenate(test_First_Conv1D_features, axis=0)
print("test_First_Conv1D_features.shape:", test_First_Conv1D_features.shape)
test_Third_Conv1D_features = np.concatenate(test_Third_Conv1D_features, axis=0)
print("test_Third_Conv1D_features.shape:", test_Third_Conv1D_features.shape)

test_Flatten_features = np.concatenate(test_Flatten_features, axis=0)
print("test_Flatten_features.shape:", test_Flatten_features.shape)

test_Linear_features = np.concatenate(test_Linear_features, axis=0)
print("test_Linear_features.shape:", test_Linear_features.shape)

labels = np.concatenate(labels, axis=0)
print("labels.shape:", labels.shape)


#input:
n,seq_len,num_size=test_inputs_features.shape
test_inputs_features=test_inputs_features.reshape((n,seq_len *num_size))
print(test_inputs_features.shape)

# 1Conv1D
n,seq_len,num_size=test_First_Conv1D_features.shape
test_First_Conv1D_features=test_First_Conv1D_features.reshape((n,seq_len *num_size))
print(test_First_Conv1D_features.shape)


# 3Conv1D
n,seq_len,num_size=test_Third_Conv1D_features.shape
test_Third_Conv1D_features=test_Third_Conv1D_features.reshape((n,seq_len *num_size ))
print(test_Third_Conv1D_features.shape)

#Flatten layer
n,seq_len,num_size=test_Flatten_features.shape
test_Flatten_features=test_Flatten_features.reshape((n,seq_len * num_size))
print(test_Flatten_features.shape)

#Linear layer
n,num_size=test_Linear_features.shape
print(test_Linear_features.shape)


# input :
tsne = TSNE(n_components=2)
test_inputs_features_tsne = tsne.fit_transform(test_inputs_features)

for i, label in enumerate(labels):
    if int(label) == 1:
        plt.scatter(test_inputs_features_tsne[i, 0], test_inputs_features_tsne[i, 1], c='r', s=0.9, facecolors='none',
                    label='Kcr')
    else:
        plt.scatter(test_inputs_features_tsne[i, 0], test_inputs_features_tsne[i, 1], c='skyblue', s=0.9,
                    facecolors='none', label='Non-Kcr')

plt.title("Input-Layer")
plt.ylabel('Dimension2', fontweight='bold')
plt.xlabel('Dimension1', fontweight='bold')


l1 = plt.Line2D(range(0), range(0), marker='o', color='r', linestyle='')
l2 = plt.Line2D(range(0), range(0), marker='o', color='skyblue', linestyle='')
plt.legend((l1, l2), ('Kcr', 'Non-Kcr'), loc='upper right', numpoints=1)
plt.savefig("../figures/Input-Layer-test-visual.jpg", dpi=600)
plt.show()

# test : First_Conv1D_features
# tsne = TSNE(n_components=2)
#
# test_First_Conv1D_stne = tsne.fit_transform(test_First_Conv1D_features)
#
# for i, label in enumerate(labels):
#     if int(label) == 1:
#         plt.scatter(test_First_Conv1D_stne[i, 0], test_First_Conv1D_stne[i, 1], c='r', s=0.9, facecolors='none',
#                     label='Kcr')
#     else:
#         plt.scatter(test_First_Conv1D_stne[i, 0], test_First_Conv1D_stne[i, 1], c='skyblue', s=0.9, facecolors='none',
#                     label='Non-Kcr')
#
# plt.title("First-Conv1D-Layer")
# plt.ylabel('Dimension2', fontweight='bold')
# plt.xlabel('Dimension1', fontweight='bold')
#
#
# l1 = plt.Line2D(range(0), range(0), marker='o', color='r', linestyle='')
# l2 = plt.Line2D(range(0), range(0), marker='o', color='skyblue', linestyle='')
# plt.legend((l1, l2), ('Kcr', 'Non-Kcr'), loc='upper right', numpoints=1)
# plt.savefig("../figures/1_Conv1D-Layer-test-visual.png", dpi=600)  # 保存图片，dpi设置分辨率
# plt.show()


# test :Third_Conv1D_features
# tsne = TSNE(n_components=2)
# test_Third_Conv1D_tsne = tsne.fit_transform(test_Third_Conv1D_features)
#
# for i, label in enumerate(labels):
#     if int(label) == 1:
#         plt.scatter(test_Third_Conv1D_tsne[i, 0], test_Third_Conv1D_tsne[i, 1], c='r', s=0.9, facecolors='none',
#                     label='Kcr')
#     else:
#         plt.scatter(test_Third_Conv1D_tsne[i, 0], test_Third_Conv1D_tsne[i, 1], c='skyblue', s=0.9,
#                     facecolors='none', label='Non-Kcr')
#
# plt.title("Third-Conv1D-Layer")
# plt.ylabel('Dimension2', fontweight='bold')
# plt.xlabel('Dimension1', fontweight='bold')
#
#
# l1 = plt.Line2D(range(0), range(0), marker='o', color='r', linestyle='')
# l2 = plt.Line2D(range(0), range(0), marker='o', color='skyblue', linestyle='')
# plt.legend((l1, l2), ('Kcr', 'Non-Kcr'), loc='upper right', numpoints=1)
#
# plt.savefig("../figures/3_Conv1D-Layer-test-visual.png", dpi=600)
# plt.show()


# test:total_outputs_features
tsne = TSNE(n_components=2)
test_total_outputs_tsne = tsne.fit_transform(test_Flatten_features)

for i, label in enumerate(labels):
    if int(label) == 1:
        plt.scatter(test_total_outputs_tsne[i, 0], test_total_outputs_tsne[i, 1], c='r', s=0.9, facecolors='none',
                    label='Kcr')
    else:
        plt.scatter(test_total_outputs_tsne[i, 0], test_total_outputs_tsne[i, 1], c='skyblue', s=0.9,
                    facecolors='none', label='Non-Kcr')

plt.title("Flatten-Layer")
plt.ylabel('Dimension2', fontweight='bold')
plt.xlabel('Dimension1', fontweight='bold')


l1 = plt.Line2D(range(0), range(0), marker='o', color='r', linestyle='')
l2 = plt.Line2D(range(0), range(0), marker='o', color='skyblue', linestyle='')
plt.legend((l1, l2), ('Kcr', 'Non-Kcr'), loc='upper right', numpoints=1)

plt.savefig("../figures/Flatten_outputs-test-visual.jpg", dpi=600)
plt.show()


# test :Linear layer
tsne = TSNE(n_components=2)
test_Linear_features_tsne = tsne.fit_transform(test_Linear_features)

for i, label in enumerate(labels):
    if int(label) == 1:
        plt.scatter(test_Linear_features_tsne[i, 0], test_Linear_features_tsne[i, 1], c='r', s=0.9, facecolors='none',
                    label='Kcr')
    else:
        plt.scatter(test_Linear_features_tsne[i, 0], test_Linear_features_tsne[i, 1], c='skyblue', s=0.9,
                    facecolors='none', label='Non-Kcr')

plt.title("Linear-Layer")
plt.ylabel('Dimension2', fontweight='bold')
plt.xlabel('Dimension1', fontweight='bold')


l1 = plt.Line2D(range(0), range(0), marker='o', color='r', linestyle='')
l2 = plt.Line2D(range(0), range(0), marker='o', color='skyblue', linestyle='')
plt.legend((l1, l2), ('Kcr', 'Non-Kcr'), loc='upper right', numpoints=1)

plt.savefig("../figures/Linear-Layer-test-visual.jpg", dpi=600)  #save fig
plt.show()


