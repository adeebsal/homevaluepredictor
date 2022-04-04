##### MOdel 2

#Clean orignal data frame and new ones to be subdivided into test/train splits
data<-na.omit(data)
rm(HouseExp_Income,TotalDebtPmt_Income,netWealth,cCredHist,
   mCredHist,pubRec,Unemply,SelfEmp,loanAppraised_Low,loanAppraised_High,
   loanAppraised_Medium,DeniedPMi,nonWhite,NotMarried,Education_yrs,denied)
# data frames
DF_D2 <- data.frame(data$denied, I(data$HouseExp_Income+data$Education_yrs), data$cCredHist,
                    data$mCredHist, data$pubRec, log(data$Unemply) ,data$SelfEmp, data$DeniedPMi, data$nonWhite, 
                    data$NotMarried, data$Education_yrs, I(data$DeniedPMi+data$DeniedPMi),
                    data$loanAppraised_High,data$loanAppraised_Low,data$loanAppraised_Medium)


# index
index <- sample(nrow(DF_D2),nrow(DF_D2)*.80)
# splits
train_DF_D2 = data[index,]
test_DF_D2 = data[-index,]

# define cost function
costfunc <- function(denied, pred_prob_train_D2) {
  weight1 <- 1
  weight0 <- 1
  c1 <- (denied==1)&(pred_prob_train_D2<optimal_cutoff)
  c0 <- (denied==0)&(pred_prob_train_D2>=optimal_cutoff)
  cost<- mean(weight1*c1+weight0*c0)
  return(cost)
}

# estimate the model
model_D2 <-glm(denied~.,data = train_DF_D2,family = binomial)


###training  
pred_prob_train_D2 <- predict.glm(model_D2,type = "response") #this is predprob for cost function
pred_train_D2 <- prediction(pred_prob_train_D2, train_DF_D2$denied)
perf_train_D2 <- performance(pred_train_D2, "tpr", "fpr")
plot(perf_train_D2,colorize=TRUE)
# AUC training set
unlist(slot(performance(pred_train_D2,"auc",),"y.values"))

# Testing 
pred_prob_test_d2 <- predict.glm(model_D2,newdata = test_DF_D2, type = "response")
pred_test_d2 <-prediction(pred_prob_test_d2,test_DF_D2$denied)
perf_test_d2 <- performance(pred_test_d2,"tpr","fpr")
plot(perf_test_d2,colorize=TRUE, add=TRUE)
# AUC Test set
unlist(slot(performance(pred_test_d2,"auc"),"y.values")) #reduction in area under curve


# determine optimal cutoff
prob_seq <- seq(0.01, 1, 0.01) 

cv_cost2= rep(0,length(prob_seq))
for (i in 1:length(prob_seq)){
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost2[i] = cv.glm(data=train_DF_D2, glmfit = model_D2, cost=costfunc, K=10)$delta[2]
} 

plot(prob_seq,cv_cost2)
optimal_cutoff_cv2 = prob_seq[which(cv_cost2==min(cv_cost2))]
optimal_cutoff_cv2 #.55.. .51

# train Classification
trainClass_D2 <-ifelse(pred_prob_train_D2>optimal_cutoff_cv2,1,0)
trainClass_D2 <- factor(trainClass_D2)
train_deny <- factor(train_DF_D2$denied)
D2train_confm <-confusionMatrix(trainClass_D2,train_deny,positive = "1")
D2train_confm



# test classification 
testclass_d2 <- ifelse(pred_prob_test_d2>optimal_cutoff_cv2,1,0)
testclass_d2 <-factor(testclass_d2)
test_deny <- factor(test_DF_D2$denied)
D2test_confm <- confusionMatrix(testclass_d2,test_deny,positive = "1")
D2test_confm











































