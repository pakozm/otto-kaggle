require(dplyr)
require(nnet)

set.seed(12345)

train <- read.csv("DATA/train.csv")
train <- train[sample(nrow(train)),]
target <- train$target
train <- data.frame(train[,2:94])

model <- multinom(target~., train, maxit=666)

test <- read.csv("DATA/test.csv")
test <- data.frame(test)

result <- predict(model, newdata=test, type='prob')
result <- data.frame(id = test$id, result)
write.csv(result, file = "result.mn.csv",row.names=FALSE, na="")
