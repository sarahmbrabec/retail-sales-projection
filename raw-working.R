############################################################
#Summary:
#This project aims to consider the relationships between retail sales and other 
#features of this dataset with the goal of creating actionable projections for the future.
#Last updated: 2/6/23
#Author: Sarah Brabec 

#packages and whatever 
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
require(xgboost)
library(caret)

#load and clean data
df <- read.csv("sales_1.csv")
str(df)
colnames(df) <- c("date","sku","group","units_pkg","avg_price_pkg","sales_pkg")
head(df)
df[is.na(df)] <- 0
df$sku <- factor(df$sku)
df$group <- factor(df$group)
summary(df)

#partition data 
#make this example reproducible
set.seed(0)
index = createDataPartition(df$sku, p = .8, list = F)
train = df[index, ]
test = df[-index, ]

train_x = data.matrix(subset(train, select = -sales_pkg))
train_y = train$sales_pkg
test_x = data.matrix(subset(test, select = -sales_pkg))
test_y = test$sales_pkg

xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#train that bitch 
watchlist = list(train=xgb_train, test=xgb_test)

model1 = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 200)

model = xgb.train(data = xgb_train, max.depth = 3,nrounds = 63, verbose = 0)

pred = predict(model, xgb_test)


mean((test_y - pred)^2) #mse
caret::MAE(test_y, pred) #mae
caret::RMSE(test_y, pred) #rmse

#Considering the results I want to try a ridge regression model. 
#XGBoost does tend to perform well, but considering the multi collinearity 
#of ridge regression would be great

library(glmnet)
#setup data
y <- df$sales_pkg
x <- data.matrix(df[, c("date","sku","group","units_pkg","avg_price_pkg")])


model <- glmnet(x, y, alpha = 0)
summary(model)

cv_model <- cv.glmnet(x, y, alpha = 0)
best_lambda <- cv_model$lambda.min
best_lambda
plot(cv_model)

best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)


plot(model, xvar = "lambda")

y_predicted <- predict(model, s = best_lambda, newx = x)

#find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

#find R-Squared
rsq <- 1 - sse/sst
rsq

#Results are absolute shit 

