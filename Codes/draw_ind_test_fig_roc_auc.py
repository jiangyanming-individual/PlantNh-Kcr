"""
draw PlantNh-Kcr  ind_test 的ROC_ACU曲线图

"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import numpy as np


ourModel_y_test_true=np.load('../np_weights/PlantNh-Kcr_y_test_true.npy').tolist()
ourModel_y_test_score=np.load('../np_weights/PlantNh-Kcr_y_test_score.npy').tolist()

fpr1,tpr1,_=roc_curve(ourModel_y_test_true,ourModel_y_test_score)
PlantNh_Kcr_roc_auc=auc(fpr1,tpr1)


plt.plot(fpr1,tpr1,label="OurModel (AUC={:.4f})".format(PlantNh_Kcr_roc_auc),lw=1,alpha=0.8,linestyle='-',color='r')


plt.plot([0,1],[0,1],lw=1,alpha=0.8,linestyle='--',color='c')

plt.xlim([-0.05, 1.05])#横竖增加一点长度 以便更好观察图像
plt.ylim([-0.05, 1.05])

plt.title('ROC curve')
plt.legend(loc=4)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')

plt.savefig('../figures/Plant-Kcr_ind_test.png')

plt.show()