##### MOdel 1

#Clean orignal data frame and new ones to be subdivided into test/train splits
data<-na.omit(data)
rm(HouseExp_Income,TotalDebtPmt_Income,netWealth,cCredHist,
   mCredHist,pubRec,Unemply,SelfEmp,loanAppraised_Low,loanAppraised_High,
   loanAppraised_Medium,DeniedPMi,nonWhite,NotMarried,Education_yrs,denied)
# data frames
DF_D1 <- data.frame(data$denied,I(data$HouseExp_Income*data$netWealth),data$cCredHist,
                    data$mCredHist,data$pubRec,data$Unemply,data$SelfEmp,data$DeniedPMi,data$nonWhite,
                    data$NotMarried,data$Education_yrs,data$DeniedPMi,I(data$Education_yrs*data$DeniedPMi))

# index
index <- sample(nrow(DF_D1),nrow(DF_D1)*.80)
# splits
train_DF_D1 = data[index,]
test_DF_D1 = data[-index,]

# define cost function
costfunc <- function(denied, pred_prob_train_D1) {
  weight1 <- 1
  weight0 <- 1
  c1 <- (denied==1)&(pred_prob_train_D1<optimal_cutoff)
  c0 <- (denied==0)&(pred_prob_train_D1>=optimal_cutoff)
  cost<- mean(weight1*c1+weight0*c0)
  return(cost)
}

# estimate model
model_D1 <-glm(denied~.,data = train_DF_D1,family = binomial)

###training  
pred_prob_train_D1 <- predict.glm(model_D1,type = "response") #this is predprob for cost function
pred_train_D1 <- prediction(pred_prob_train_D1, train_DF_D1$denied)
perf_train_D1 <- performance(pred_train_D1, "tpr", "fpr")
plot(perf_train_D1,colorize=TRUE)
# AUC training set
unlist(slot(performance(pred_train_D1,"auc",),"y.values"))


# Testing 
pred_prob_test_d1 <- predict.glm(model_D1,newdata = test_DF_D1, type = "response")
pred_test_d1 <-prediction(pred_prob_test_d1,test_DF_D1$denied)
perf_test_d1 <- performance(pred_test_d1,"tpr","fpr")

plot(perf_test_d1,colorize=TRUE, add=TRUE)
# AUC Test set
unlist(slot(performance(pred_test_d1,"auc"),"y.values")) #reduction in area under curve



prob_seq <- seq(0.01, 1, 0.01) 

cv_cost1= rep(0,length(prob_seq))
for (i in 1:length(prob_seq)){
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost1[i] = cv.glm(data=train_DF_D1, glmfit = model_D1, cost=costfunc, K=10)$delta[2]
} 

plot(prob_seq,cv_cost1)
optimal_cutoff_cv1 = prob_seq[which(cv_cost1==min(cv_cost1))]
optimal_cutoff_cv1 #.58

# train Classification
trainClass_D1 <-ifelse(pred_prob_train_D1>optimal_cutoff_cv1,1,0)
trainClass_D1 <- factor(trainClass_D1)
train_deny <- factor(train_DF_D1$denied)
D1train_confm <-confusionMatrix(trainClass_D1,train_deny,positive = "1")
D1train_confm

# test classification 
testclass_d1 <- ifelse(pred_prob_test_d1>optimal_cutoff_cv1,1,0)
testclass_d1 <-factor(testclass_d1)
test_deny <- factor(test_DF_D1$denied)
D1test_confm <- confusionMatrix(testclass_d1,test_deny,positive = "1")
D1test_confm

