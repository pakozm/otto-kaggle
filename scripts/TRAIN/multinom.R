require(dplyr)
require(nnet)

train <- read.csv("DATA/train.csv")
train <- train[sample(nrow(train)),]
target <- train[1:100,]$target
train <- data.frame(train[1:100,2:94])

model <- multinom(target~., train, maxit=666)

test <- read.csv("DATA/test.csv")
test <- data.frame(test[1:100,2:94])

result <- predict(model, newdata=test, type='prob')
result <- data.frame(result)

print(result)

write.table(result, file = "result.R.csv",row.names=TRUE, na="",col.names=TRUE, sep=",")
