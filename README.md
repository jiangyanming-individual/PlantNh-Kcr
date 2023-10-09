# PlantNh-Kcr
This code is for the article 'PlantNh-Kcr: A deep learning model for predicting non-histone crotonylation sites in plants'
Please feel free to contact me if you need any further information and help, email : jym19943856480@163.com.

###The introduction of each folder
1. We used python version 3.9 and pytorch version 1.13.1.

2. **The CML_classifer folder** and **CML_weights folder** contain conventional machine learning classifiers(for example, RF, AdaBoost, and LightGBM) and the model weight information saved during model training and independent testing, respectively.

3. The code for the five-fold cross-validation, independent test, and image visualization is located in **the Codes folder**. the five-fold cross-validation model weights and independent test model weight are located in **the model_weights folder**, which also contains the model weights for each plant.

4. **The Csv folder** contains the training and test sets of various plants divided at 7:3 ratio.

5. There are the files of training sets and independent test sets in **the Datasets folder**, which also include the datasets after Cd-hit with 40% sequence identity.

6. **The DL_classifer folder** and **DL_weights folder** contain deep learning classifiers(for example, CNN, LSTM, and BiLSTM) and the model weight information saved during model training and independent testing, respectively.

7. **The figures folder** includes the images for this paper.

8. The information on metric values(such as Sn, Sp, ACC, MCC, and AUC) is located in **the np_weights folder**.