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
iphonedata <- read.csv("iphone_smallmatrix_labeled_8d.csv")
galaxydata <- read.csv("galaxy_smallmatrix_labeled_8d.csv")

# protect raw data
iphonedata <- iphone.small.matrix

galaxydata <- galaxy.small.matrix

### ---- Feature Selection & Preprocessing ----
## Removing rows with no iphone and galaxy observations
# Column 1-5 represents the number of instances that type of phone mentioned in a webpage
# and take only iphone reviews because ios reviews are creating noise.
iphonedata %>%
  filter(iphone > 0) %>% 
  select(starts_with("iphone"),
         -iphone) -> iphonedata

# Take the sentiment column out of the data
iphone.sent <- iphonedata[ncol(iphonedata)]
iphone.vars <- iphonedata[-ncol(iphonedata)]

# Sum the total reviews
iphone.vars$rowsums <- rowSums((iphone.vars))
summary(iphone.vars$rowsums)

# Bind the sentiment column
iphone.df <- bind_cols(iphone.vars, iphone.sent)

# Remove rows with zero variance, change this number to alter number of rows
iphone.df <- iphone.df %>% filter(rowsums > 5)

# Remove the unnecessary column
iphone.df$rowsums <- NULL

# Take only samsung reviews because google android reviews are irrelevant.
galaxydata %>%
  filter(samsunggalaxy > 0) %>% 
  select(starts_with("samsung"), 
         starts_with("galaxy"),
         -samsunggalaxy) -> galaxydata

# Take the sentiment column out of the data
galaxy.sent <- galaxydata[ncol(galaxydata)]
galaxy.vars <- galaxydata[-ncol(galaxydata)]

# Sum the total reviews
galaxy.vars$rowsums <- rowSums((galaxy.vars))
summary(galaxy.vars$rowsums)

# Bind the sentiment column
galaxy.df <- bind_cols(galaxy.vars, galaxy.sent)

# Remove rows with zero variance, change this number to alter number of rows
galaxy.df %>% filter(rowsums > 1) -> galaxy.df

# Remove the unnecessary column
galaxy.df$rowsums <- NULL

## Check balance of data
# iphone data
iphone.df %>% 
  group_by(iphonesentiment) %>% 
  summarise(count(iphonesentiment))

# galaxy data
galaxy.df %>% 
  group_by(galaxysentiment) %>% 
  summarise(count(galaxysentiment))

## Recode sentiment to combine factor levels
iphone.df$iphonesentiment <- recode(iphone.df$iphonesentiment, 
                                    "0" = "N", 
                                    "1" = "N", 
                                    "2" = "N", 
                                    "3" = "P", 
                                    "4" = "P", 
                                    "5" = "P") 

galaxy.df$galaxysentiment <- recode(galaxy.df$galaxysentiment, 
                                    "0" = "N", 
                                    "1" = "N", 
                                    "2" = "N", 
                                    "3" = "P", 
                                    "4" = "P", 
                                    "5" = "P") 

## Change dependent variables' data type
iphone.df$iphonesentiment <- as.factor(iphone.df$iphonesentiment)
galaxy.df$galaxysentiment <- as.factor(galaxy.df$galaxysentiment)
str(iphone.df)
str(galaxy.df)

### ---- Sampling & Data Partition ----
## iphone sampling
set.seed(1234)
iphonedata.both <- ovun.sample(iphonesentiment~., 
                               data = iphone.df, 
                               N = nrow(iphone.df),
                               p = 0.5, 
                               seed = 1, 
                               method = "both")$data

iphonedata.both %>% 
  group_by(iphonesentiment) %>% 
  summarise(count(iphonesentiment))

## iphone data partition
intrain1 <- createDataPartition(y = iphonedata.both$iphonesentiment, 
                                p = 0.7, 
                                list = FALSE)
iphonetrain <- iphonedata.both[intrain1,]
iphonetest <- iphonedata.both[-intrain1,]

## galaxy sampling
set.seed(2345)
galaxydata.both <- ovun.sample(galaxysentiment~., 
                               data = galaxy.df,
                               N = nrow(galaxy.df), 
                               p = 0.5, 
                               seed = 1, 
                               method = "both")$data

galaxydata.both %>% 
  group_by(galaxysentiment) %>% 
  summarise(count(galaxysentiment))

## galaxy data partition
intrain2 <- createDataPartition(y = galaxydata.both$galaxysentiment, 
                                p = 0.7, 
                                list = FALSE)
galaxytrain <- galaxydata.both[intrain2,]
galaxytest <- galaxydata.both[-intrain2,]

### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores.
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

### ---- Random Forest Modelization ----
set.seed(4567)
## Set Train Control and Grid
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 10,
                         preProc = c("center", "scale"),
                         repeats = 1,
                         verboseIter = TRUE)

## iphone
RFmodel1 <- train(iphonesentiment ~ ., 
                  iphonetrain,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel1

plot(RFmodel1)

plot(varImp(RFmodel1))

predRFmodel1 <- predict(RFmodel1, iphonetest)

postResample(predRFmodel1, iphonetest$iphonesentiment) -> RFmodel1metrics

RFmodel1metrics

cmRFiphone <- confusionMatrix(predRFmodel1, iphonetest$iphonesentiment) 
cmRFiphone

## galaxy
RFmodel2 <- train(galaxysentiment ~ ., 
                  galaxytrain,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel2

plot(RFmodel2)

varImp(RFmodel2)

predRFmodel2 <- predict(RFmodel2, galaxytest)

postResample(predRFmodel2, galaxytest$galaxysentiment) -> RFmodel2metrics

RFmodel2metrics

cmRFgalaxy <- confusionMatrix(predRFmodel2, galaxytest$galaxysentiment) 
cmRFgalaxy
