data<-na.omit(data)
rm(HouseExp_Income,TotalDebtPmt_Income,netWealth,cCredHist,
   mCredHist,pubRec,Unemply,SelfEmp,loanAppraised_Low,loanAppraised_High,
   loanAppraised_Medium,DeniedPMi,nonWhite,NotMarried,Education_yrs,denied)
# data frames
DF_D3 <- data.frame(data$denied, data$HouseExp_Income, data$TotalDebtPmt_Income, data$netWealth, data$cCredHist,
                    data$mCredHist, data$Unemply ,data$SelfEmp, data$loanAppraised_Low , data$loanAppraised_Medium,
                    data$nonWhite,  data$loanAppraised_High,data$DeniedPMi,data$nonWhite,
                    data$NotMarried, data$Education_yrs)
#index
index <- sample(nrow(DF_D3),nrow(DF_D3)*.80)
# splits
train_DF_D3 = data[index,]

# cost function
costfunc <- function(denied, pred_prob_train_D3) {
  weight1 <- 1
  weight0 <- 1
  c1 <- (denied==1)&(pred_prob_train_D3<optimal_cutoff)
  c0 <- (denied==0)&(pred_prob_train_D3>=optimal_cutoff)
  cost<- mean(weight1*c1+weight0*c0)
  return(cost)
}


### Determining which poly values to use
v1 <- numeric(4)
v2 <- numeric(4)
v3 <- numeric(4)
v4 <- numeric(4)
polys <-array(c(v1,v2,v3,v4), dim=c(4,4,4,4))

for (i in 1:4){
  for (j in 1:4){
    for (k in 1:4){
      for (l in 1:4){
        model_d3 <- glm(denied~ HouseExp_Income + poly(TotalDebtPmt_Income,i,raw=TRUE) + netWealth + 
                          poly(cCredHist,j,raw=TRUE) + mCredHist + poly(Unemply,k,raw=TRUE) + SelfEmp + loanAppraised_Low + loanAppraised_Medium 
                        + loanAppraised_High + DeniedPMi  + nonWhite + NotMarried + poly(Education_yrs,l,raw=TRUE),
                        data = train_DF_D3, family = binomial)
        pred_prob_train_D3 <- predict.glm(model_d3,type = "response") #this is predprob for cost function
        pred_train_D3 <- prediction(pred_prob_train_D3, train_DF_D3$denied)
        perf_train_D3 <- performance(pred_train_D3, "tpr", "fpr")
        # AUC training set
        polys[i,j,k,l] <- unlist(slot(performance(pred_train_D3,"auc",),"y.values"))
      }
    }
  }
}
polySet <-  which(polys == max(polys),arr.ind=TRUE)
polySet
# MAX AUC .834 for the poly values 4 3 4 3
print(polys[polySet])



## Hardcoding a data frame with those poly values 

DF_D3a <- data.frame(data$denied, data$HouseExp_Income, I(data$TotalDebtPmt_Income**4), data$netWealth, I(data$cCredHist**3),
                    data$mCredHist, I(data$Unemply**4) ,data$SelfEmp, data$loanAppraised_Low , data$loanAppraised_Medium,
                    data$nonWhite,  data$loanAppraised_High,data$DeniedPMi,data$nonWhite,
                    data$NotMarried, I(data$Education_yrs**4))

# Split new frame                
train_DF_D3a = data[index,]
test_DF_D3a = data[-index,]

# model estimation
model_D3a <-glm(denied~.,data = train_DF_D3a,family = binomial)

###training  
pred_prob_train_D3a <- predict.glm(model_D3a,type = "response") #this is predprob for cost function
pred_train_D3a <- prediction(pred_prob_train_D3a, train_DF_D3a$denied)
perf_train_D3a <- performance(pred_train_D3a, "tpr", "fpr")
plot(perf_train_D3a,colorize=TRUE)
# AUC training set
unlist(slot(performance(pred_train_D3a,"auc",),"y.values")) # .82865

# Testing 
pred_prob_test_d3a <- predict.glm(model_D3a,newdata = test_DF_D3a, type = "response")
pred_test_d3a <-prediction(pred_prob_test_d3a,test_DF_D3a$denied)
perf_test_d3a <- performance(pred_test_d3a,"tpr","fpr")
plot(perf_test_d3a,colorize=TRUE, add=TRUE)
# AUC Test set
unlist(slot(performance(pred_test_d3a,"auc"),"y.values")) #reduction in area under curve # .8434

# determine optimal cutoff
prob_seq <- seq(0.01, 1, 0.01) 

cv_cost3a= rep(0,length(prob_seq))
for (i in 1:length(prob_seq)){
  optimal_cutoff = prob_seq[i]
  set.seed(123)
  cv_cost3a[i] = cv.glm(data=train_DF_D3a, glmfit = model_D3a, cost=costfunc, K=10)$delta[2]
} 

## Optimal Cutoff Score
plot(prob_seq,cv_cost3a)
optimal_cutoff_cv3a = prob_seq[which(cv_cost3a==min(cv_cost3a))]
optimal_cutoff_cv3a #.41

# train Classification
trainClass_D3a <-ifelse(pred_prob_train_D3a>optimal_cutoff_cv3a,1,0)
trainClass_D3a <- factor(trainClass_D3a)
train_deny <- factor(train_DF_D3a$denied)
D3atrain_confm <-confusionMatrix(trainClass_D3a,train_deny,positive = "1")
D3atrain_confm

# test classification 
testclass_d3a <- ifelse(pred_prob_test_d3a>optimal_cutoff_cv3a,1,0)
testclass_d3a <-factor(testclass_d3a)
test_deny <- factor(test_DF_D3a$denied)
D3atest_confm <- confusionMatrix(testclass_d3a,test_deny,positive = "1")
D3atest_confm

























