#######################
#Sentiment prediction
#
#By Julian Ariza
#
######################

#Libraries
library(doParallel)
library(caret)
library(ggplot2)
library(dplyr)
library(plotly)

#Loading the data
iphonedata <- readRDS("iphone_dataframe.rds")

galaxydata <- readRDS("galaxy_dataframe.rds")

#Preprocessing

#Data types 
iphonedata$iphonesentiment <- as.factor(iphonedata$iphonesentiment)
galaxydata$galaxysentiment <- as.factor(galaxydata$galaxysentiment)


#Partitioning the data

# iphone
set.seed(6378)
iphonesample <- iphonedata[sample(1:nrow(iphonedata),
                                  1000, 
                                  replace = FALSE), ]
intrain1 <- createDataPartition(y = iphonesample$iphonesentiment, 
                                p = 0.7, 
                                list = FALSE)
iphonetrain <- iphonesample[intrain1,]
iphonetest <- iphonesample[-intrain1,]

## galaxy
set.seed(2345)
intrain2 <- createDataPartition(y = galaxydata$galaxysentiment, 
                                p = 0.7, 
                                list = FALSE)
galaxytrain <- galaxydata[intrain2,]
galaxytest <- galaxydata[-intrain2,]

#Cores selection and cluster registration

cl <- makeCluster(3)

registerDoParallel(cl)


#Model with random forest algorithm

## Set Train Control and Grid
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 2)

RFgrid <- expand.grid(mtry=c(1:5))

## iphone
RFmodel1 <- train(iphonesentiment ~ ., 
                  iphonetrain,
                  method = "rf",
                  trControl = RFtrctrl,
                  tuneGrid = RFgrid,
                  tuneLenght = 2)

RFmodel1

plot(RFmodel1)
#checking the variables relationship
varImp(RFmodel1)
#Checking the prediction with validation data
predRFmodel1 <- predict(RFmodel1, iphonetest)

postResample(predRFmodel1, iphonetest$iphonesentiment) -> RFmodel1metrics

RFmodel1metrics

cmRFiphone <- confusionMatrix(predRFmodel1, iphonetest$iphonesentiment) 
cmRFiphone

## galaxy
RFmodel2 <- train(galaxysentiment ~ ., 
                  galaxytrain,
                  method = "rf",
                  trControl = RFtrctrl,
                  tuneGrid = RFgrid,
                  tuneLenght = 2)

RFmodel2

plot(RFmodel2)
#checking the variables relationship
varImp(RFmodel2)
#Checking the results with validation data
predRFmodel2 <- predict(RFmodel2, galaxytest)

postResample(predRFmodel2, galaxytest$galaxysentiment) -> RFmodel2metrics

RFmodel2metrics

cmRFgalaxy <- confusionMatrix(predRFmodel2, galaxytest$galaxysentiment) 
cmRFgalaxy


