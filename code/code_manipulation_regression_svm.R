#### by Ni Ma 20170925
#### Read data
listings<-read.csv("~/Documents/datathon/NYC Datathon Materials/listings.csv", header =T,na.strings=c(""))
zipcodes = unique(listings$zipcode)

dataset = as.data.frame(matrix(nrow = length(zipcodes), ncol = 0))
dataset$zipcode = zipcodes
for (i in 1: length(zipcodes)) {
    prices = listings$price[which(listings$zipcode == zipcodes[i])]
    prices = as.numeric(sub('\\$','',as.character(prices)))
    mean_price = mean(prices, na.rm = TRUE)
    number_price = length(prices)
    dataset$mean_price[i] = mean_price
    dataset$number_price[i] = number_price
    dataset$metropolitan[i] = as.character(listings$metropolitan[which(listings$zipcode == zipcodes[i])[1]])
    dataset$state[i] = as.character(listings$state[which(listings$zipcode == zipcodes[i])[1]])
}

dataset = dataset[-which(is.na(dataset$zipcode)),]
write.csv(dataset,"~/Documents/datathon/listings_combined.csv")


real_estate<-read.csv("~/Documents/datathon/NYC Datathon Materials/real_estate.csv", header =T,na.strings=c(""))
zipcodes = dataset$zipcode
for (i in 1: length(zipcodes)) {
    ZHVI_rank = real_estate$size_rank[which(real_estate$zipcode == zipcodes[i] & real_estate$type == 'ZHVI')]
    ZRI_rank = real_estate$size_rank[which(real_estate$zipcode == zipcodes[i] & real_estate$type == 'ZRI')]
    ZHVI_latest_value = real_estate$X2017.06[which(real_estate$zipcode == zipcodes[i] & real_estate$type == 'ZHVI')]
    city = as.character(real_estate$city[which(real_estate$zipcode == zipcodes[i])[1]])
    if (length(ZHVI_rank)>0) {
    dataset$ZHVI_rank[i] = ZHVI_rank
    }
    if (length(ZRI_rank)) {
    dataset$ZRI_rank[i] = ZRI_rank
    }
    if (length(ZHVI_latest_value)) {
    dataset$ZHVI_latest_value[i] = ZHVI_latest_value
    }
    if (length(city)) {
        dataset$city[i] = city
    }
}

write.csv(dataset,"~/Documents/datathon/listings_combined_real_estate.csv")


data_all<-read.csv("~/Documents/datathon/data_all.csv", header =T,na.strings=c(""))
indexes = sample(1:nrow(data_all), size=floor(0.75*nrow(data_all)))
data_train = data_all[indexes,]
data_test = data_all[-indexes,]

write.csv(data_train,"~/Documents/datathon/train_data.csv")
write.csv(data_test,"~/Documents/datathon/test_data.csv")



###regression
data_train<-read.csv("~/Documents/datathon/train_data_price.csv", header =T,na.strings=c(""))
data_test<-read.csv("~/Documents/datathon/test_data_price.csv", header =T,na.strings=c(""))

for (name in names(data_train)) {
    data_train[,name] = as.numeric(data_train[,name])
    data_test[,name] = as.numeric(data_test[,name])
}

for (name in names(data_train)[-1]) {
    data_train[,name] = data_train[, name]/sd(data_train[, name])
    data_test[,name] = data_test[, name]/sd(data_test[, name])
}

###ols
ols = lm(mean_price ~ ZHVI_rank+ZHVI_latest_value+X2016Q3_gdp+X2016Q3_per_capita+X75.84_years+X.9.999_or_less, data = data_train)
####
Call:
lm(formula = mean_price ~ ZHVI_rank + ZHVI_latest_value + X2016Q3_gdp +
X2016Q3_per_capita + X75.84_years + X.9.999_or_less, data = data_train)

Residuals:
Min      1Q  Median      3Q     Max
-1.2934 -0.5972 -0.1425  0.3142  7.8269

Coefficients:
Estimate Std. Error t value Pr(>|t|)
(Intercept)         1.39494    0.29278   4.764 2.79e-06 ***
ZHVI_rank           0.26464    0.05923   4.468 1.07e-05 ***
ZHVI_latest_value   0.24989    0.05559   4.495 9.48e-06 ***
X2016Q3_gdp         1.02870    1.08505   0.948 0.343754
X2016Q3_per_capita -1.23715    1.06812  -1.158 0.247558
X75.84_years        0.19368    0.06899   2.808 0.005274 **
X.9.999_or_less    -0.22846    0.06416  -3.561 0.000421 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.9339 on 347 degrees of freedom
Multiple R-squared:  0.1426,    Adjusted R-squared:  0.1277
F-statistic: 9.616 on 6 and 347 DF,  p-value: 8.573e-10
####
mse.train <- sqrt(summary(ols)$sigma^2)
#74.76781

test.pred <- predict(ols,newdata=data_test)
test.y    <- data_test$mean_price
mse.test  <- sqrt(mean((test.pred - test.y)^2))
#95.68859

###ridge
#install.packages("glmnet")
#library(glmnet)
y= as.matrix(data_train$mean_price)
x = as.matrix(data_train[,c('ZHVI_rank', 'ZHVI_latest_value', 'X2016Q3_gdp', 'X2016Q3_per_capita', 'X75.84_years', 'X.9.999_or_less')])
y_test= as.matrix(data_test$mean_price)
x_test = as.matrix(data_test[,c('ZHVI_rank', 'ZHVI_latest_value', 'X2016Q3_gdp', 'X2016Q3_per_capita', 'X75.84_years', 'X.9.999_or_less')])

ridge <- glmnet(x, y, alpha = 0)
cv.out <- cv.glmnet(x, y, alpha = 0)
bestlam <- cv.out$lambda.min
ridge.pred <- predict(ridge, s = bestlam, newx = x_test)

ridge.train <- predict(ridge, s = bestlam, newx = x)

sqrt(mean((ridge.train-y)^2))
#74.21548
sqrt(mean((ridge.pred-y_test)^2))
#94.70719
###lasso

lasso <- glmnet(x, y, alpha = 1)
lasso.train <- predict(lasso, s = bestlam, newx = x)
lasso.pred <- predict(lasso, s = bestlam, newx = x_test)
sqrt(mean((lasso.train-y)^2))
#76.15058
sqrt(mean((lasso.pred-y_test)^2))
#94.42657
mean((lasso.pred-ytest)^2)




####svm
#install.packages("e1071")
#library(e1071)

svm <- svm(mean_price ~ ., data = data_train)
svm.pred <- predict(svm, data_test[,-1])
svm.train<- predict(svm, data_train[,-1])

sqrt(mean((svm.train-data_train$mean_price)^2))
sqrt(mean((svm.pred-data_test$mean_price)^2))
#75.19135
#98.96103

svm <- svm(mean_price ~ ZHVI_rank + ZHVI_latest_value + X2016Q3_gdp +
X2016Q3_per_capita + X75.84_years + X.9.999_or_less, data = data_train)
svm.pred <- predict(svm, data_test[,-1])
svm.train<- predict(svm, data_train[,-1])

sqrt(mean((svm.train-data_train$mean_price)^2))
sqrt(mean((svm.pred-data_test$mean_price)^2))
#75.0117
#96.31665

write.csv(ridge.train,"~/Documents/datathon/ridge_train_result.csv")
write.csv(ridge.pred,"~/Documents/datathon/ridge_test_result.csv")



