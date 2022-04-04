
# @author: Cody Olivotto
# R-Code

#### Libraries ####
library(boot)
library(caret)
library(ROCR)
library(dplyr)
#import orignal Dataset
hmda_data <- read.csv("c:/ucf_classes/eco_4444_fall_2021/data/hmda_sw.txt", sep = '\t', header = TRUE)
attach(hmda_data)

####################### create variables #######################################
# Note: these variables created in R, are exported and reused in python 

HouseExp_Income <- ifelse(s45>30.0,1,0)
TotalDebtPmt_Income <- s46
netWealth <- netw
cCredHist <- s43 
mCredHist <- s42
pubRec <- s44
Unemply <- uria
SelfEmp <- s27a
loanAppraised_Low <- ifelse(s6/s50<=.8,1,0)
loanAppraised_Medium <- ifelse((.95>s6/s50 & s6/s50>.80),1,0)
loanAppraised_High <- ifelse(s6/s50>.95, 1,0)
DeniedPMi <- s53
TwoToFourFamily <- dprop # (Omitted) The reduced data set only focuses on single family homes 
nonWhite <- ifelse(s13==5,0,1)

denied <-  ifelse((s7==3),1,0)
#approved <- ifelse(denied==1,0,1)
#check <- data.frame(approved,denied)



# an 'approved' dummy variable was created,but was omitted for easier comparability with full set in pdf

################################ Prep the Data ###################################

# Data needed for the model in a cleaned up frame & Randomized
data <- data.frame(denied,HouseExp_Income,TotalDebtPmt_Income,netWealth,cCredHist,mCredHist,pubRec,Unemply,SelfEmp,  
                   loanAppraised_Low, loanAppraised_Medium, loanAppraised_High, DeniedPMi, nonWhite)

#### Test/Train Split
set.seed(123)
index <- sample(nrow(data),nrow(data)*.80)
train = data[index,]
test = data[-index,]
######################### Linear Model ########################################

###### Linear model Training
lm <- glm(denied~., data = train)
summary(lm)

lm_predprob_train <- predict.glm(lm, type = "response")

lm_pred_train <- prediction(lm_predprob_train, train$denied)
lm_perf_train <- performance(lm_pred_train, "tpr", "fpr")

plot(lm_perf_train, colorize=TRUE)
unlist(slot(performance(lm_pred_train,"auc"), "y.values")) #training performance .8234

##### Testing Linear Model

lm_predprob_test <- predict.glm(lm,newdata = test, type="response")

lm_pred_test <- prediction(lm_predprob_test, test$denied)
lm_perf_test <- performance(lm_pred_test, "tpr", "fpr")

plot(lm_perf_test, colorize=TRUE, add=TRUE)
# TEST PERFROMANCE AUC
unlist(slot(performance(lm_pred_test, "auc"), "y.values"))

######################### Linear Comparisons #####################################

lmCoef <- 0
for (i in 1:14){
  lmCoef[i]<- lm$coefficients[i]
}
LM_table2_HMDA <- c(-0.22, 0.06, 0.005, 0.000004, 0.04, 0.03, 0.19, 0.01, 0.05, -0.12, -0.05, 0.10, 0.65, 0.07) 
LM_Comparison <- data.frame(lmCoef,LM_table2_HMDA)

# For comparison a data.frame was created with our estimated lm coefficients in the left
# most column, followed by the results from table 2 in the hmda pdf
LM_Comparison

# Psuedo R2 from Glm
actual <- test$denied
predy <- predict(lm, newdata = test)

lm_RSS <- sum((actual-predy)^2)
lm_Var <- sum((actual-mean(actual))^2)
lm_R2 <- (1- lm_RSS/lm_Var)
lm_psuedo_adjR2 <- 1-(((1-lm_R2)*(length(actual)-1))/(length(actual)-(length(ls(data))-1)-1))
lm_psuedo_adjR2 # estimated value: .2085      HMDA table 2 value: .32

######################### Logit Model ########################################

### Training Logit model
logit <- glm(denied~., data = train, family = binomial)
summary(logit)

logit_predprob_train <- predict.glm(logit, type = "response")

logit_pred_train <- prediction(logit_predprob_train, train$denied)
logit_perf_train <- performance(logit_pred_train, "tpr", "fpr")

plot(logit_perf_train, colorize=TRUE)
unlist(slot(performance(logit_pred_train,"auc"), "y.values")) #training performance .825

##### Testing Logit Model

logit_predprob_test <- predict.glm(logit,newdata = test, type="response")

logit_pred_test <- prediction(logit_predprob_test, test$denied)
logit_perf_test <- performance(logit_pred_test, "tpr", "fpr")

plot(logit_perf_test, colorize=TRUE, add=TRUE)
# TEST PERFROMANCE AUC
unlist(slot(performance(logit_pred_test, "auc"), "y.values")) # test performance .833

######################### logit Comparisons #####################################

logitCoef <- 0
for (i in 1:14){
  logitCoef[i]<- logit$coefficients[i]
}
logit_table2_HMDA <- c(-13.69, 0.63, 0.08, 0.00008, 0.51, 0.43, 1.95, 0.11 , .70 , -0.89 , 0.13 , 1.40, 6.16, 1.00)
Logit_Comparison <-data.frame(logit_table2_HMDA,logitCoef)
Logit_Comparison

######################### Final R comparison (LM and Logit w/ HMDA Coef Values) ##################################

# Full Comparison format for columns: Estimated outer values, and hmda provided inner values
moniker <- c("Intercept", "HouseExp_Income","TotalDebtPmt_Income","netWealth","cCredHist","mCredHist","pubRec","Unemply","SelfEmp",  
             "loanAppraised_Low", "loanAppraised_Medium", "loanAppraised_High", "DeniedPMi", "nonWhite")
Comparison <- data.frame(LM_Comparison,moniker, Logit_Comparison)

Comparison  

###################### export data for DF for use in R ###########################
write.csv(data,"c:/ucf_classes/eco_4444_fall_2021/hw7rData.csv", row.names = FALSE)
write.csv(train,"c:/ucf_classes/eco_4444_fall_2021/hw7rDataTrain.csv", row.names = FALSE)
write.csv(test,"c:/ucf_classes/eco_4444_fall_2021/hw7rDataTest.csv", row.names = FALSE)
write.csv(LM_table2_HMDA,"c:/ucf_classes/eco_4444_fall_2021/hw7lmHDMAcoef.csv", row.names = FALSE)
write.csv(logit_table2_HMDA,"c:/ucf_classes/eco_4444_fall_2021/hw7logitHDMAcoef.csv", row.names = FALSE)
