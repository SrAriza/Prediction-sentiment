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
library(e1071)
library(randomForest)
library(ROSE)

#Loading the data
iphonedata <- readRDS("iphone_dataframe.rds")

galaxydata <- readRDS("galaxy_dataframe.rds")


# Change data types
iphonedata$iphonesentiment <- as.factor(iphonedata$iphonesentiment)
galaxydata$galaxysentiment <- as.factor(galaxydata$galaxysentiment)


#Create a new dataset that will be used for recoding sentiment
iphoneRC <- iphonedata

galaxyRC <- galaxydata

## Recode sentiment to combine factor levels
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, 
                                   "VN" = "N", 
                                   "N" = "N", 
                                   "SN" = "N", 
                                   "VP" = "P", 
                                   "P" = "P", 
                                   "SP" = "P") 

galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, 
                                   "VN" = "N", 
                                   "N" = "N", 
                                   "SN" = "N", 
                                   "VP" = "P", 
                                   "P" = "P", 
                                   "SP" = "P") 

## Change dependent variable data type
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)

galaxyRC$galaxysentiment <- as.factor(galaxyRC$galaxysentiment)


# Sampling  
# iphone undersampling
set.seed(4635)
iphonedata.under <- ovun.sample(iphonesentiment~., 
                                data = iphoneRC, 
                                p = 0.5, 
                                seed = 1, 
                                method = "under")$data

iphonedata.under %>% 
  group_by(iphonesentiment) %>% 
  summarise(count(iphonesentiment))

## galaxy oversampling
set.seed(2345)
galaxydata.over <- ovun.sample(galaxysentiment~., 
                               data = galaxyRC, 
                               p = 0.5, 
                               seed = 1, 
                               method = "over")$data

galaxydata.over %>% 
  group_by(galaxysentiment) %>% 
  summarise(count(galaxysentiment))

# Core Selection
# Find how many cores are on your machine
detectCores() 

# Create cluster with desired number of cores.
cl <- makeCluster(3)

# Register cluster
registerDoParallel(cl)


#Principal Component Analysis 
# iphone
preprocessParamsiphone <- preProcess(iphonedata.under[,-14], 
                                     method=c("center", "scale", "pca"), 
                                     thresh = 0.95)
print(preprocessParamsiphone)

# use predict to apply pca parameters, create training, exclude dependant
iphone.pca <- predict(preprocessParamsiphone, iphonedata.under[,-14])

# add the dependent to training
iphone.pca$iphonesentiment <- iphonedata.under$iphonesentiment

# inspect results
str(iphone.pca)

## galaxy
preprocessParamsgalaxy <- preProcess(galaxydata.over[,-14], 
                                     method=c("center", "scale", "pca"), 
                                     thresh = 0.95)
print(preprocessParamsgalaxy)

# use predict to apply pca parameters, create training, exclude dependant
galaxy.pca <- predict(preprocessParamsgalaxy, galaxydata.over[,-14])

# add the dependent to training
galaxy.pca$galaxysentiment <- galaxydata.over$galaxysentiment

# inspect results
str(galaxy.pca)

### ---- Recursive Feature Elimination ----
## Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

## Use rfe and omit the response variable (attribute 8 iphonesentiment & 5 galaxysentiment) 
rfeResults1 <- rfe(iphone.pca[,1:7], 
                   iphone.pca$iphonesentiment, 
                   sizes = (1:7), 
                   rfeControl = ctrl)

rfeResults2 <- rfe(galaxy.pca[,1:4], 
                   galaxy.pca$galaxysentiment, 
                   sizes = (1:4), 
                   rfeControl = ctrl)

## Get results
rfeResults1
predictors(rfeResults1)

rfeResults2
predictors(rfeResults2)

## Plot results
plot(rfeResults1, type=c("g", "o"))

plot(rfeResults2, type=c("g", "o"))

## Create new data set with rfe recommended features
iphoneRFE <- iphone.pca[,predictors(rfeResults1)]

galaxyRFE <- galaxy.pca[,predictors(rfeResults2)]

## Add the dependent variable to iphoneRFE and galaxyRFE
iphoneRFE$iphonesentiment <- iphone.pca$iphonesentiment

galaxyRFE$galaxysentiment <- galaxy.pca$galaxysentiment



# Data Partition
# iphone data partition
intrain1 <- createDataPartition(y = iphoneRFE$iphonesentiment, 
                                p = 0.7, 
                                list = FALSE)
iphonetrain <- iphoneRFE[intrain1,]
iphonetest <- iphoneRFE[-intrain1,]

## galaxy data partition
intrain2 <- createDataPartition(y = galaxyRFE$galaxysentiment, 
                                p = 0.7, 
                                list = FALSE)
galaxytrain <- galaxyRFE[intrain2,]
galaxytest <- galaxyRFE[-intrain2,]

# Random Forest Modelization 

# Set Train Control and Grid
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 2)

# iphone
RFmodel1 <- train(iphonesentiment ~ ., 
                  iphonetrain,
                  method = "rf",
                  trControl = RFtrctrl,
                  tuneLenght = 2)

RFmodel1

plot(RFmodel1)

varImp(RFmodel1)

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
                  tuneLenght = 2)

RFmodel2

plot(RFmodel2)

varImp(RFmodel2)

predRFmodel2 <- predict(RFmodel2, galaxytest)

postResample(predRFmodel2, galaxytest$galaxysentiment) -> RFmodel2metrics

RFmodel2metrics

cmRFgalaxy <- confusionMatrix(predRFmodel2, galaxytest$galaxysentiment) 
cmRFgalaxy
