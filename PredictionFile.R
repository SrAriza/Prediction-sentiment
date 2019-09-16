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

#Create Cluster with desired number of cores.  
cl <- makeCluster(3)
registerDoParallel(cl)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)

