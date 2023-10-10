
"""
Compare with other non-histone models
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


nhKcr_y_test_true=np.load('../np_weights/nhKcr_y_test_true.npy').tolist()
nhKcr_y_test_score=np.load('../np_weights/nhKcr_y_test_score.npy').tolist()

CaspNhKcr_y_test_true=np.load('../np_weights/CapsMh-Kcr_y_true.npy').tolist()
CapsNhKcr_y_test_score=np.load('../np_weights/CapsNh-Kcr_y_pred_score.npy').tolist()

ourModel_y_test_true=np.load('../np_weights/PlantNh-Kcr_y_test_true.npy').tolist()
ourModel_y_test_score=np.load('../np_weights/PlantNh-Kcr_y_test_score.npy').tolist()



fpr1,tpr1,_=roc_curve(ourModel_y_test_true,ourModel_y_test_score)
PlantNh_Kcr_roc_auc=auc(fpr1,tpr1)

fpr2,tpr2,_=roc_curve(CaspNhKcr_y_test_true,CapsNhKcr_y_test_score)
CpasNhKcr_roc_auc=auc(fpr2,tpr2)

fpr3,tpr3,_=roc_curve(nhKcr_y_test_true,nhKcr_y_test_score)
nhKcr_roc_auc=auc(fpr3,tpr3)


plt.plot(fpr3,tpr3,label="nhKcr (AUC={:.4f})".format(nhKcr_roc_auc),lw=1,alpha=0.8,linestyle='-',color='hotpink')
plt.plot(fpr2,tpr2,label="CapsNh-Kcr (AUC={:.4f})".format(CpasNhKcr_roc_auc),lw=1,alpha=0.8,linestyle='-',color='b')
plt.plot(fpr1,tpr1,label="PlantNh-Kcr (AUC={:.4f})".format(PlantNh_Kcr_roc_auc),lw=1,alpha=0.8,linestyle='-',color='r')


plt.plot([0,1],[0,1],lw=1,alpha=0.8,linestyle='--',color='c')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.title('ROC curve')
plt.legend(loc=4)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')

plt.savefig('../figures/compare_nh-models.png')

plt.show()