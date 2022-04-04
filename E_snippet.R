# -------------------- E. Evaluating the superior model -----------------------

#fresh data
data <- na.omit(data)

DF_final <- data.frame(data) #insert required variables
index <- sample(nrow(DF_final),nrow(DF_final)*.80)
# splits
train = data[index,]
test = data[-index,]


bestModel <- glm() #hardcode best model

