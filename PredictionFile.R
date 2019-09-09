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


cl <- makeCluster(3)
