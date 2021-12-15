# COMS4995_Project
## Fraud Detection in Credit Card transactions data
<center><i>Chenhui Mao, Sriram Dommeti, Yewen Zhou, Yeqi Zhang, Zhifeng Zhang</i></center>

### Introduction
Credit Card transaction frauds are an increasing problem. 
Fraud detection in credit card transactions data is essential for consumer banks or the credit card company, 
to help in preventing fraud loss, 
saving millions of dollars. In this analysis, 
we will be using publicly accessible dataset[https://github.com/CapitalOneRecruiting/DS] of credit card transactions provided by Capital One company, for exploratory data analysis and to build predictive models for flagging a transaction as fraud or non fraud. Dataset has a target variable indicating whether a transaction is a fraud or not fraud. The features in the data set include transaction details, account details, merchant details like expenditure category, card details, card usage type, location data, etc. Challenge here is that data is highly imbalanced.


### Exploratory Data Analysis
Code for EDA is under visualization folder

### Model
We have trained 5 different models: Neural Network, Random Forest, GBDT, XGBoost, SVM.
Training code for Neural Network can be seen in the train.ipynb.\\
Training code for Random Forest can be seen in the model/Random Forest.ipynb\\
Training code for XGBoost can be seen in the model/XGBoost performance tuning.ipynb\\
Training code for GBDT can be seen in the 