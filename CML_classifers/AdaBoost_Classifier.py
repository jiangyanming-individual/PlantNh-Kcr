# author:Lenovo
# datetime:2023/7/22 19:00
# software: PyCharm
# project:PlantNh-Kcr


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,auc,roc_auc_score,roc_curve
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from collections import Counter
import torch.nn as nn
import torch


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
    #
    y=np.array(y)
    print(y.shape)

    return X,y


def get_EGAAC_encoding(data):

    X=[]
    y=[]

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
        # print(counter)

        for key in groupKeys:

            for aa in group[key]:
                groupCount_dict[key]=groupCount_dict.get(key,0)+counter[aa]

        for key in groupKeys:
            one_code.append(round(groupCount_dict[key] / len(seq), 3))
        X.append(one_code)
        y.append(int(label))

    X=np.array(X)
    n,dim=X.shape #(n,5)
    #
    # # reshape
    print("new X shape :",X.shape)
    #
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
        AAindex_names.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)

    props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(':')

    if props:
        tempAAindex_names = []
        tempAAindex = []

        for p in props:
            if AAindex_names.index(p) != -1:
                tempAAindex_names.append(p)
                tempAAindex.append(AAindex[AAindex_names.index(p)])


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
                for aaindex in AAindex:
                    one_code.append(0)
                continue
            for aaindex in AAindex:

                one_code.append(float(aaindex[seq_index.get(aa)]))
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
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X
    }


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

        X.append(one_code.detach().cpu().numpy())
        y.append(int(label))

    X=np.array(X)
    n,seq_len,dim=X.shape

    # reshape
    X=np.reshape(X,(n,seq_len * dim))
    print("new X shape :",X.shape)

    y=np.array(y)
    print(y.shape)

    return X,y



train_params=[]

train_SN=[]
train_SP=[]
train_ACC=[]
train_MCC=[]


def AdaBoost_Classifer(train_data,ind_test_data):


    ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)

    X_train,y_train=train_data

    kf=KFold(n_splits=5,shuffle=True)

    fold=1

    y_true=[]
    y_score=[]

    for train_index,valid_index in kf.split(X_train,y_train):

        TP, FP, TN, FN = 0, 0, 0, 0
        print("第{}次交叉验证开始...".format(fold))
        # print(train_index)
        # print(valid_index)
        #
        this_train_x,this_train_y=X_train[train_index],y_train[train_index]

        # print(this_train_x)
        # print(this_train_y)
        #
        this_valid_x,this_valid_y=X_train[valid_index],y_train[valid_index]

        ada_clf.fit(this_train_x,this_train_y)

        # train fit
        y_train_pred=ada_clf.predict(this_train_x)

        # valid fit
        y_valid_pred=ada_clf.predict(this_valid_x)

        #acc:
        train_acc = accuracy_score(this_train_y, y_train_pred)
        valid_acc = accuracy_score(this_valid_y, y_valid_pred)

        print("训练集准确率: {:.2f}%".format(train_acc * 100))
        print("验证集准确率: {:.2f}%".format(valid_acc * 100))

        #acu:
        y_true.append(this_valid_y)
        y_score.append(ada_clf.predict_proba(this_valid_x)[:,1])
        # print("y_score:",y_score)

        # 5Kfold SN、SP、ACC、MCC

        y_valid_true_label=this_valid_y
        y_valid_pred_label=y_valid_pred


        res=confusion_matrix(y_valid_true_label,y_valid_pred_label)
        print("混淆矩阵:",res)

        TP += ((y_valid_true_label == 1) & (y_valid_pred_label == 1)).sum().item()
        FP += ((y_valid_true_label == 0) & (y_valid_pred_label == 1)).sum().item()
        TN += ((y_valid_true_label == 0) & (y_valid_pred_label == 0)).sum().item()
        FN += ((y_valid_true_label == 1) & (y_valid_pred_label == 0)).sum().item()

        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        train_SN.append(SN)
        train_SP.append(SP)
        train_ACC.append(ACC)
        train_MCC.append(MCC)

        print("Train TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("Train SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))

        fold+=1

    train_params.append(np.mean(train_SN))
    train_params.append(np.mean(train_SP))
    train_params.append(np.mean(train_ACC))
    train_params.append(np.mean(train_MCC))

    np.save("../CML_weights/Ada_5kfold_BLOSUM62_params.npy", train_params)
    print("Train Mean : SN is {},SP is {},ACC is {},MCC is {}".format(np.mean(train_SN), np.mean(train_SP), np.mean(train_ACC),np.mean(train_MCC)))
    print("ind_test start ...")


    TP, FP, TN, FN = 0, 0, 0, 0
    y_true=np.concatenate(y_true,axis=0)
    y_score=np.concatenate(y_score,axis=0)

    fpr,tpr,_=roc_curve(y_true,y_score)
    valid_auc=auc(fpr,tpr)
    print("valid auc :",valid_auc)

    X_test, y_test = ind_test_data
    y_test_pred=ada_clf.predict(X_test)
    test_acc=accuracy_score(y_test,y_test_pred)

    print("测试集准确率 {:.2f}:".format(test_acc * 100))
    # ind test auc:
    test_auc=roc_auc_score(y_test,ada_clf.predict_proba(X_test)[:,1])

    np.save('../CML_weights/Ada_BLOSUM62_y_test_true.npy', y_test)
    np.save('../CML_weights/Ada_BLOSUM62_y_test_score.npy', ada_clf.predict_proba(X_test)[:, 1])

    print("test auc :",test_auc)

    #calculate SN、SP、ACC、MCC

    y_test_true_label=y_test
    y_test_pred_label=y_test_pred


    TP += ((y_test_true_label == 1) & (y_test_pred_label == 1)).sum().item()
    FP += ((y_test_true_label == 0) & (y_test_pred_label == 1)).sum().item()
    TN += ((y_test_true_label == 0) & (y_test_pred_label == 0)).sum().item()
    FN += ((y_test_true_label == 1) & (y_test_pred_label == 0)).sum().item()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))


    test_params=[]

    test_params.append(SN)
    test_params.append(SP)
    test_params.append(ACC)
    test_params.append(MCC)

    #save test SN、SP、ACC、MCC
    np.save("../CML_weights/Ada_test_BLOSUM62_params.npy", test_params)

    print("ind test: TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind test: SN is {},SP is {},ACC is {},MCC is {}".format(SN, SP, ACC, MCC))


if __name__ == '__main__':

    train=read_file(train_path)
    ind_test=read_file(ind_test_path)


    #binary encode:
    # train_data=get_Binary_encoding(train)
    # ind_test_data=get_Binary_encoding(ind_test)
    # AdaBoost_Classifer(train_data,ind_test_data)



    #AAC encode:
    # train_data=get_AAC_encoding(train)
    # ind_test_data=get_AAC_encoding(ind_test)
    # AdaBoost_Classifer(train_data,ind_test_data)

    #EGAAC encode
    # train_data=get_EGAAC_encoding(train)
    # ind_test_data=get_EGAAC_encoding(ind_test)
    # AdaBoost_Classifer(train_data,ind_test_data)


    #AAindex encode
    # train_data=get_AAindex_encode(train)
    # ind_test_data=get_AAindex_encode(ind_test)
    # AdaBoost_Classifer(train_data,ind_test_data)


    #BLOSUM62 encode：
    train_data=get_BLOSUM62_encoding(train)
    ind_test_data=get_BLOSUM62_encoding(ind_test)
    AdaBoost_Classifer(train_data,ind_test_data)


