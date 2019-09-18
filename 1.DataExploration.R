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

iphone <- read.csv("iphone_smallmatrix_labeled_8d.csv")
galaxy <- read.csv("galaxy_smallmatrix_labeled_8d.csv")

#Create Cluster with desired number of cores.  
cl <- makeCluster(3)
registerDoParallel(cl)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)

#Plots
plot_ly(iphone, x= ~iphonesentiment, type='histogram')
plot_ly(galaxy, x= ~galaxysentiment, type='histogram')

#Checking for na´s:
any(is.na(iphone))
any(is.na(galaxy))

#Removing values without iphone or galaxy:
iphone %>%
  filter(iphone != 0) %>% 
  select(starts_with("ios"), starts_with("iphone")) -> iphone.df

galaxy %>%
  filter(samsunggalaxy != 0) %>% 
  select(starts_with("google"), starts_with("samsung"), starts_with("galaxy")) -> galaxy.df

#increase max print
options(max.print=1000000)

## Check correlations
cor(iphone.df)

cor(galaxy.df)

#Changing data type
iphone.df$iphonesentiment <- as.factor(iphone.df$iphonesentiment)

galaxy.df$galaxysentiment <- as.factor(galaxy.df$galaxysentiment)


# Sample the data before using RFE
set.seed(6424)
iphone.sample <- iphone.df[sample(1:nrow(iphone.df),
                                  1000, 
                                  replace = FALSE), ]

## Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

## Use rfe and omit the response variable (attribute 15 iphonesentiment & galaxysentiment) 
rfeResults1 <- rfe(iphone.sample[,1:14], 
                   iphone.sample$iphonesentiment, 
                   sizes = (1:14), 
                   rfeControl = ctrl)

rfeResults2 <- rfe(galaxy.df[,1:14], 
                   galaxy.df$galaxysentiment, 
                   sizes = (1:14), 
                   rfeControl = ctrl)

# Get results
#Results iphone
rfeResults1
predictors(rfeResults1)
#Results galaxy
rfeResults2
predictors(rfeResults2)

# Plot results
plot(rfeResults1, type=c("g", "o"))
plot(rfeResults2, type=c("g", "o"))

# New dataset with recommended features
iphoneRFE <- iphone.df[,predictors(rfeResults1)]

galaxyRFE <- galaxy.df[,predictors(rfeResults2)]


# Add the dependent variable to iphoneRFE & galaxyRFE
iphoneRFE$iphonesentiment <- iphone.df$iphonesentiment

galaxyRFE$galaxysentiment <- galaxy.df$galaxysentiment

# Review outcome
str(iphoneRFE)

str(galaxyRFE)

#Rename Levels of Factor ----
iphoneRFE %>%
  mutate(
    iphone.sentiment = 
      case_when(iphonesentiment %in% "0" ~ "VN",
                iphonesentiment %in% "1" ~ "N",
                iphonesentiment %in% "2" ~ "SN",
                iphonesentiment %in% "3" ~ "SP",
                iphonesentiment %in% "4" ~ "P",
                iphonesentiment %in% "5" ~ "VP")) -> iphoneRFE2

iphoneRFE2$iphonesentiment <- NULL

names(iphoneRFE2)[names(iphoneRFE2) == "iphone.sentiment"] <- "iphonesentiment"

galaxyRFE %>%
  mutate(
    galaxy.sentiment = 
      case_when(galaxysentiment %in% "0" ~ "VN",
                galaxysentiment %in% "1" ~ "N",
                galaxysentiment %in% "2" ~ "SN",
                galaxysentiment %in% "3" ~ "SP",
                galaxysentiment %in% "4" ~ "P",
                galaxysentiment %in% "5" ~ "VP")) -> galaxyRFE2

galaxyRFE2$galaxysentiment <- NULL

names(galaxyRFE2)[names(galaxyRFE2) == "galaxy.sentiment"] <- "galaxysentiment"

### ---- Save Datasets for Modelization ----
saveRDS(iphoneRFE2, file = "iphone_dataframe.rds")

saveRDS(galaxyRFE2, file = "galaxy_dataframe.rds")

## Stop Cluster
stopCluster(cl)