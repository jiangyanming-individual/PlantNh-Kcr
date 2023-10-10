"""
draw CML的ROC_ACU曲线图

"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import numpy as np


RF_BE_y_test_true=np.load('../CML_weights/RF_BE_y_test_true.npy').tolist()
RF_BE_y_test_score=np.load('../CML_weights/RF_BE_y_test_score.npy').tolist()


Ada_AAindex_y_test_true=np.load('../CML_weights/Ada_AAindex_y_test_true.npy').tolist()
Ada_AAindex_y_test_score=np.load('../CML_weights/Ada_AAindex_y_test_score.npy').tolist()

LightGBM_BLOSUM62_y_test_true=np.load('../CML_weights/LGB_BLOSUM62_y_test_true.npy').tolist()
LightGBM_BLOUSUM62_y_test_score=np.load('../CML_weights/LGB_BLOSUM62_y_test_score.npy').tolist()

LSTM_BE_y_test_true=np.load('../np_weights/LSTM(BE)_y_test_true.npy').tolist()
LSTM_BE_y_test_score=np.load('../np_weights/LSTM(BE)_y_test_score.npy').tolist()


BiLSTM_BE_y_test_true=np.load('../np_weights/BiLSTM(BE)_y_test_true.npy').tolist()
BiLSTM_BE_y_test_score=np.load('../np_weights/BiLSTM(BE)_y_test_score.npy').tolist()


CNN_BE_y_test_true=np.load('../np_weights/CNN(BE)_y_test_true.npy').tolist()
CNN_BE_y_test_score=np.load('../np_weights/CNN(Be)_y_test_score.npy').tolist()

#
ourModel_y_test_true=np.load('../np_weights/PlantNh-Kcr_y_test_true.npy').tolist()
ourModel_y_test_score=np.load('../np_weights/PlantNh-Kcr_y_test_score.npy').tolist()



#calculate roc_auc
fpr1,tpr1,_=roc_curve(RF_BE_y_test_true,RF_BE_y_test_score)
RF_BE_roc_auc=auc(fpr1,tpr1)


fpr2,tpr2,_=roc_curve(Ada_AAindex_y_test_true,Ada_AAindex_y_test_score)
Ada_AAindex_roc_auc=auc(fpr2,tpr2)


fpr3,tpr3,_=roc_curve(LightGBM_BLOSUM62_y_test_true,LightGBM_BLOUSUM62_y_test_score)
LightGBM_BLOSUM62_roc_auc=auc(fpr3,tpr3)

fpr4,tpr4,_=roc_curve(LSTM_BE_y_test_true,LSTM_BE_y_test_score)
LSTM_BE_roc_auc=auc(fpr4,tpr4)


fpr5,tpr5,_=roc_curve(BiLSTM_BE_y_test_true,BiLSTM_BE_y_test_score)
BiLSTM_BE_roc_auc=auc(fpr5,tpr5)


fpr6,tpr6,_=roc_curve(CNN_BE_y_test_true,CNN_BE_y_test_score)
CNN_AAindex_roc_auc=auc(fpr6,tpr6)


fpr7,tpr7,_=roc_curve(ourModel_y_test_true,ourModel_y_test_score)
PlantNh_Kcr_roc_auc=auc(fpr7,tpr7)


plt.plot(fpr1,tpr1,label="RF(BE) (AUC={:.4f})".format(RF_BE_roc_auc),lw=1,alpha=0.8,linestyle='-',color='green')
plt.plot(fpr2,tpr2,label="AdaBoost(AAindex) (AUC={:.4f})".format(Ada_AAindex_roc_auc),lw=1,alpha=0.8,linestyle='-',color='mediumslateblue')
plt.plot(fpr3,tpr3,label="LightGBM(BLOSUM62) (AUC={:.4f})".format(LightGBM_BLOSUM62_roc_auc),lw=1,alpha=0.8,linestyle='-',color='dodgerblue')
plt.plot(fpr4,tpr4,label="LSTM(BE) (AUC={:.4f})".format(LSTM_BE_roc_auc),lw=1,alpha=0.8,linestyle='-',color='orange')
plt.plot(fpr5,tpr5,label="BiLSTM(BE) (AUC={:.4f})".format(BiLSTM_BE_roc_auc),lw=1,alpha=0.8,linestyle='-',color='cornflowerblue')
plt.plot(fpr6,tpr6,label="CNN(BE) (AUC={:.4f})".format(CNN_AAindex_roc_auc),lw=1,alpha=0.8,linestyle='-',color='b')
plt.plot(fpr7,tpr7,label="PlantNh-Kcr (AUC={:.4f})".format(PlantNh_Kcr_roc_auc),lw=1,alpha=0.8,linestyle='-',color='r')

plt.plot([0,1],[0,1],lw=1,alpha=0.8,linestyle='--',color='c')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.title('ROC curve')
plt.legend(loc=4,fontsize=9)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')

plt.savefig('../figures/compare_models.png')

plt.show()
