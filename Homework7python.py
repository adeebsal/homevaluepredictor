# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:02:09 2021
Finished on Fri Dec 3 18:52:48 2021
@author: Cody Olivotto
"""


"""
Note: the data was all seperated and repared in R. The only actions taken on this 
 are the regressions, and their performance was measured in terms of being able to predict 
 the occurence of an applicant being denied
"""

################### Libraries ###############################################
from pandas import read_csv
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn import metrics


# Improt the preformed variables and data.frames from R
# Note this data is alredy seperated into the variables used in table 2 and
# is randomized and seperated into test and training sets

# full set of x & y used 
data = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7rData.csv", sep = ',')

# split to test/train
train = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7rDataTrain.csv", sep = ',', )
test = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7rDataTest.csv", sep = ',')

# Split to test x, test y
testY = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7rDataTest.csv", sep = ',', usecols=[0])
testX = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7rDataTest.csv", sep = ',')
testX = testX.drop(testX.columns[0], axis=1) # running this line more than one time reduce your predictors

# Split to train x, test y
trainY = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7rDataTrain.csv", sep = ',', usecols=[0])
trainX = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7rDataTrain.csv", sep = ',', )
trainX = trainX.drop(trainX.columns[0], axis=1) # running this line more than one time reduce your predictors

# Import the hardcoded hmda table 2 values to compare w/ estimates
hmdaTable2_lm = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7lmHDMAcoef.csv", sep = ',')
hmdaTable2_logit = read_csv("c:/ucf_classes/eco_4444_fall_2021/hw7logitHDMAcoef.csv", sep = ',')


"""
#########################The Linear Model    #################################
"""
# Sklearn used to determine  AUC
lm_reg = LinearRegression()
lm = lm_reg.fit(trainX, trainY)

lm_pred_prob = lm_reg.predict(testX)
fpr, tpr, _ = metrics.roc_curve(testY, lm_pred_prob)
lm_auc = metrics.roc_auc_score(testY, lm_pred_prob)

# smf.ols used to obtain clean output for coef
lm1 = smf.ols('denied ~ HouseExp_Income + TotalDebtPmt_Income + netWealth + cCredHist + mCredHist + pubRec + Unemply + SelfEmp + loanAppraised_Low + loanAppraised_Medium + loanAppraised_High + DeniedPMi + nonWhite', data = train)
lm_results = lm1.fit()
LmCoef = lm_results.params

"""
#### Results for Linear Model ###
"""
# the area under curve for Lm
lm_auc # 83.26 %

# these allow you to compare coefficents
LmCoef # Obtained Values
hmdaTable2_lm # Values given in the HMDA PDF

"""
######################## Logit Model ########################################
"""
# Sklearn was creating to many errors with logistic regression, due the solver "‘lbfgs’"
# so the whole process is demonstrated below with stats models, which obtained similar
# results to our R-coded models, and also showed a slight increase to the AUC

# Stats models logit Regression
logit = smf.logit('denied ~ HouseExp_Income + TotalDebtPmt_Income + netWealth + cCredHist + mCredHist + pubRec + Unemply + SelfEmp + loanAppraised_Low + loanAppraised_Medium + loanAppraised_High + DeniedPMi + nonWhite', data = train)
logit = logit.fit()
print(logit.summary())
LogitCoef = logit.params

logit_pred_prob = logit.predict(testX)
fpr, tpr, _= metrics.roc_curve(testY, logit_pred_prob)
logit_auc = metrics.roc_auc_score(testY, logit_pred_prob)

"""
#### Results for Logit Model ###
"""
# the area under curve for Lm
logit_auc # 83.38%

#compare coefficent estimates 

LogitCoef # Obtained Values
hmdaTable2_logit # Values given in hmda pdf









