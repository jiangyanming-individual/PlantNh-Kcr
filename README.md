# PlantNh-Kcr
This reposiory is for the article 'PlantNh-Kcr: A deep learning model for predicting non-histone crotonylation sites in plants'. 
The code was implemented in python 3.9 and pytorch 1.13.1. Note that  if the program has a memory overflow problem occurs, please try setting **device='cpu'**.
Please feel free to contact me if you need any further information and help, by email jym19943856480@163.com.


### The introduction of each folder is as below

1. **The CML_classifer folder** and **CML_weights folder** contain conventional machine learning classifiers(for example, RF, AdaBoost, and LightGBM) and the model weight information saved during model training and independent testing.


2. The code for the five-fold cross-validation, independent test are located in **the Codes folder**. The weights of models on five-fold cross-validation model weight and independent test, and the weights of models for each plant are located in **the model_weights folder**.


4. There are the files of total training set and independent tes set in **the Datasets folder**.


3. **The Species_train_test_sets** contains the csv files of training and test sets for each plant.
   

5. **The np_weights folder**  contains the metrics value results of the model.
