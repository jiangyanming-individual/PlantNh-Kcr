# author:Lenovo
# datetime:2023/7/22 18:55
# software: PyCharm
# project:PlantNh-Kcr


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,auc,roc_auc_score,roc_curve
from numpy import interp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.utils import shuffle
import warnings
import math
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from collections import Counter
import torch.nn as nn
import torch
warnings.filterwarnings("ignore")

"""
Binary encode
"""

AA_Seq='ACDEFGHIKLMNPQRSTVWYX'

train_path="../Datasets/train.csv"
ind_test_path="../Datasets/ind_test.csv"

def read_file(filepath):

    data=[]
    with open(filepath,mode='r',encoding='utf-8') as f:

        for line in f.readlines():
            seq,label=line.strip().split(',')
            data.append((seq,label))
        f.close()
    return data

def get_Binary_encoding(data):

    X=[]
    y=[]
    for seq,label in data:
        one_code=[]
        for i in seq:
            vector=[0]*21
            vector[AA_Seq.index(i)]=1
            one_code.append(vector)
        X.append(one_code)
        y.append(int(label))
    X=np.array(X)
    n,seq_len,dim=X.shape
    # reshape
    X=np.reshape(X,(n,seq_len * dim))
    print("new X shape :",X.shape)
    y=np.array(y)
    print(y.shape)
    return X,y


def get_AAC_encoding(data):

    X=[]
    y=[]
    for seq,label in data:
        one_code=[]
        counter=Counter(seq)
        # print(counter)
        for key in counter:
            # 计算概率
            counter[key] = round(counter[key] / len(seq), 3)
        for item in AA_Seq:
            one_code.append(counter[item])
        # print(one_code)
        X.append(one_code)
        y.append(int(label))
    X=np.array(X)
    n,dim=X.shape
    print("new X shape :",X.shape)
    y=np.array(y)
    print(y.shape)
    return X,y


def get_EGAAC_encoding(data):

    X=[]
    y=[]
    """分为5组"""
    group = {
        'Aliphatic group': 'GAVLMI',
        'Aromatic groups': 'FYW',
        'Positively charged groups': 'KRH',
        'Negatively charged groups': 'DE',
        'No charge group': 'STCPNQ'
    }
    groupKeys = group.keys()

    for seq,label in data:
        one_code=[]
        groupCount_dict = {}
        counter=Counter(seq)
        for key in groupKeys:
            #遍历每一组:统计每组的个数
            for aa in group[key]:
                groupCount_dict[key]=groupCount_dict.get(key,0)+counter[aa]
        #计算每组的概率：
        for key in groupKeys:
            one_code.append(round(groupCount_dict[key] / len(seq), 3))
        # print("one_code:",one_code)
        X.append(one_code)
        y.append(int(label))
    X=np.array(X)
    n,dim=X.shape #(n,5)
    #reshape
    print("new X shape :",X.shape)
    y=np.array(y)
    print(y.shape)

    return X,y


def get_AAindex_encode(data):

    X=[]
    y=[]
    with open('../features_encode/AAindex/AAindex_normalized.txt', mode='r') as f:
        records=f.readlines()[1:]
        f.close()
    AA = 'ARNDCQEGHILKMFPSTWYV'

    AAindex_names = []
    AAindex = []
    for i in records:
        # print(i.rstrip().split()[0]) #得到AAindex的names
        AAindex_names.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
    props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(':')

    if props:
        tempAAindex_names = []
        tempAAindex = []
        for p in props:
            # 如果29种的一种存在
            if AAindex_names.index(p) != -1:
                tempAAindex_names.append(p)
                tempAAindex.append(AAindex[AAindex_names.index(p)])
        # 如果找到了，就将前29种的性质直接替代AAindx；
        if len(tempAAindex_names) != 0:
            AAindex_names = tempAAindex_names
            AAindex = tempAAindex
    seq_index = {} #(0-19)
    for i in range(len(AA)):
        seq_index[AA[i]] = i

    for seq,label in data:
        one_code=[]
        for aa in seq:
            if aa == 'X':
                for aaindex in AAindex:  # 为X 全部赋值为0
                    one_code.append(0)
                continue
            for aaindex in AAindex:
                # print(type(aaindex[seq_index.get(aa)]))
                one_code.append(aaindex[seq_index.get(aa)])  # 添加存在的aaindex;
        X.append(one_code) #(29,29)
        # print(one_code)
        y.append(int(label))
    X=np.array(X)
    n,seq_len=X.shape
    print("new X shape :",X.shape)
    y=np.array(y)
    print(y.shape)
    return X,y


def get_BLOSUM62_encoding(data):

    X=[]
    y=[]
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }

    # 对blosum62取均值：
    for key in blosum62:
        for index,value in enumerate(blosum62[key]):
            blosum62[key][index]=round((value + 4) / 15,3)
    for seq,label in data:
        one_code=[]
        for aa in seq:
            one_code.extend(blosum62.get(aa)) #(29,21)
        X.append(one_code)
        y.append(int(label))
    X=np.array(X)
    print("X shape:",X.shape)
    y=np.array(y)
    print(y.shape)
    return X,y


# WordEmbedding:
def get_WordEmbedding_encoding(data):

    X=[]
    y=[]
    AA = 'ARNDCQEGHILKMFPSTWYVX'

    seq_index = {} #(0-20)
    for i in range(len(AA)):
        seq_index[AA[i]] = i
    for seq,label in data:

        one_code=[]
        for i in seq:
            one_code.append(seq_index.get(i))
        word_Embedding=nn.Embedding(num_embeddings=len(AA),embedding_dim=5)
        one_code=torch.tensor(one_code)
        one_code=word_Embedding(one_code)

        # print(one_code)
        X.append(one_code.detach().cpu().numpy())
        y.append(int(label))

    X=np.array(X)
    # print(X.shape)
    n,seq_len,dim=X.shape
    # reshape
    X=np.reshape(X,(n,seq_len * dim))
    print("new X shape :",X.shape)

    y=np.array(y)
    print(y.shape)

    return X,y


#train
train_SN = []
train_SP = []
train_ACC = []
train_F1_score = []
train_MCC = []
train_AUC = []

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

    print(
        "ind test Mean metrics : SN is {:.3f},SP is {:.3f},ACC is {:.3f},F1-score is {:.3f},MCC is {:.3f},AUC is {:.3f}".
        format(mean_SN, mean_SP, mean_ACC, mean_F1_score, mean_MCC, mean_AUC))
    print(
        "ind test std metrics : SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".
        format(std_SN, std_SP, std_ACC, std_F1_score, std_MCC, std_AUC))

def cross_validation(RF_model,X_train,y_train):
    kf = KFold(n_splits=5, shuffle=True)
    fold = 1
    for train_index, valid_index in kf.split(X_train, y_train):
        print("第{}次交叉验证开始...".format(fold))
        this_train_x, this_train_y = X_train[train_index], y_train[train_index]
        this_valid_x, this_valid_y = X_train[valid_index], y_train[valid_index]
        RF_model.fit(this_train_x, this_train_y)
        # train fit
        y_train_pred = RF_model.predict(this_train_x)
        # valid fit
        y_valid_pred = RF_model.predict(this_valid_x)
        # acc:
        train_acc = accuracy_score(this_train_y, y_train_pred)
        valid_acc = accuracy_score(this_valid_y, y_valid_pred)
        print("训练集准确率: {:.2f}%".format(train_acc * 100))
        print("验证集准确率: {:.2f}%".format(valid_acc * 100))

        # 5Kfold SN、SP、ACC、F1_score,MCC
        y_valid_true_label = this_valid_y
        y_valid_score=RF_model.predict_proba(this_valid_x)[:, 1]
        y_valid_pred_label = y_valid_pred

        print("混淆矩阵")
        (TN, TP, FN, FP), (SN, SP, ACC, MCC, F1Score) = Calculate_confusion_matrix(y_valid_true_label,
                                                                                   y_valid_pred_label)
        valid_auc=roc_auc_score(y_valid_true_label,y_valid_score)
        #save metrics:
        train_SN.append(SN)
        train_SP.append(SP)
        train_ACC.append(ACC)
        train_F1_score.append(F1Score)
        train_MCC.append(MCC)
        train_AUC.append(valid_auc)

        print("Train TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("Train SN is {},SP is {},ACC is {},F1-score is {}, MCC is {}, AUC is {}".format(SN, SP, ACC, F1Score, MCC,valid_auc))

        fold += 1

    print(
        "Train Mean metrics values: SN is {:.3f},SP is {:.3f},ACC is {:.3f},F1-score is {:.3f},MCC is {:.3f},AUC is {:.3f}".format(np.mean(train_SN),
                                                                                   np.mean(train_SP),
                                                                                   np.mean(train_ACC),
                                                                                   np.mean(train_F1_score),
                                                                                   np.mean(train_MCC),
                                                                                   np.mean(train_AUC)))
    print("Train Std metrics values : SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".format(np.std(train_SN),
                                                                                    np.std(train_SP),
                                                                                    np.std(train_ACC),
                                                                                    np.std(train_F1_score),
                                                                                    np.std(train_MCC),
                                                                                    np.std(train_AUC)))

def independent_test(RF_model,X_train,y_train,X_test,y_test,random_seed):

    print("-----------------------------ind_test start------------------------")
    X_train,y_train=shuffle(X_train,y_train,random_state=random_seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)
    #fit
    RF_model.fit(X_train, y_train)
    #predict
    y_test_pred_label= RF_model.predict(X_test)
    # ind test auc:
    y_test_score=RF_model.predict_proba(X_test)[:, 1]
    # calculate SN、SP、ACC、MCC,F1score
    y_test_true_label = y_test
    y_test_pred_label = y_test_pred_label
    fpr, tpr, _ = roc_curve(y_test, y_test_score)
    test_auc = metrics.auc(fpr, tpr) #y_test_true, y_test_score
    print("test auc :", test_auc)
    # mean tpr and fpr
    tpr = interp(mean_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

    (TN, TP, FN, FP), (SN, SP, ACC, MCC, F1Score) = Calculate_confusion_matrix(y_test_true_label, y_test_pred_label)
    total_SN.append(SN)
    total_SP.append(SP)
    total_ACC.append(ACC)
    total_F1_score.append(F1Score)
    total_MCC.append(MCC)
    total_AUC.append(test_auc)

    print("ind test: TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind test: SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".format(SN, SP, ACC, F1Score, MCC,test_auc))

def RF_Classifer(train_data,ind_test_data):

    rf_clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_leaf_nodes=10, min_impurity_decrease=0.0, bootstrap=True,
                                oob_score=False, n_jobs=1, random_state=42, verbose=1,
                                warm_start=False, class_weight='balanced')

    X_train, y_train = train_data
    X_test, y_test = ind_test_data
    # cross validation:
    cross_validation(rf_clf, X_train, y_train)
    # ind test:
    random_seed = 42
    for i in range(10):
        np.random.seed(random_seed)
        rf_clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10,
                                        min_samples_split=2, min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0,
                                        max_leaf_nodes=10, min_impurity_decrease=0.0, bootstrap=True,
                                        oob_score=False, n_jobs=1, random_state=random_seed, verbose=1,
                                        warm_start=False, class_weight='balanced')
        random_seed += 10
        independent_test(rf_clf, X_train, y_train, X_test, y_test, random_seed)
    Calcuate_mean_std_metrics_values(total_SN, total_SP, total_ACC, total_F1_score, total_MCC, total_AUC)
    #save mean tprs and fprs
    mean_tpr=np.mean(tprs,axis=0)
    np.save('../CML_weights/RF_AAindex_test_mean_fpr.npy', mean_fpr)
    np.save('../CML_weights/RF_AAindex_test_mean_tpr.npy', mean_tpr)
    np.save('../CML_weights/RF_AAindex_test_AUCs.npy', total_AUC)
    print("mean AUC value is {:.3f}".format(metrics.auc(mean_fpr,mean_tpr)))
    print("mean AUC value is {:.3f}".format(np.mean(total_AUC)))
    print("std AUC values is {:.4f}".format(np.std(total_AUC)))

if __name__ == '__main__':

    # ind_test
    total_SN = []
    total_SP = []
    total_ACC = []
    total_F1_score = []
    total_MCC = []
    total_AUC = []#add AUC

    mean_fpr = np.linspace(0, 1, 101)
    mean_fpr[-1] = 1.0
    tprs = []

    train=read_file(train_path)
    ind_test=read_file(ind_test_path)

    #binary encode:
    # train_data=get_Binary_encoding(train)
    # ind_test_data=get_Binary_encoding(ind_test)
    # RF_Classifer(train_data,ind_test_data)

    #AAC encode:
    # train_data=get_AAC_encoding(train)
    # ind_test_data=get_AAC_encoding(ind_test)
    # RF_Classifer(train_data,ind_test_data)

    #EGAAC encode
    # train_data=get_EGAAC_encoding(train)
    # ind_test_data=get_EGAAC_encoding(ind_test)
    # RF_Classifer(train_data,ind_test_data)

    #AAindex encode
    train_data=get_AAindex_encode(train)
    ind_test_data=get_AAindex_encode(ind_test)
    RF_Classifer(train_data,ind_test_data)

    #BLOSUM62 encode：
    # train_data=get_BLOSUM62_encoding(train)
    # ind_test_data=get_BLOSUM62_encoding(ind_test)
    # RF_Classifer(train_data,ind_test_data)


