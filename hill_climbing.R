library(readxl)
library(readr)
library(bnlearn)

data_set <- read_csv("E:/College/3. Third Year/SEM V/2. Labs/CS367 Artificial Inteliigence/Traffic-Flow-Prediction/Data/TrafficDataSet.csv",  col_select = c("Time", "Dayoftheweek", "CarCount", "BikeCount", "BusCount", "TruckCount", "TrafficSituation"))



data2 <- lapply(data_set, as.factor)
data3 <- as.data.frame(data2)
pdag <- hc(data3)
print(pdag)

plot(pdag)

fitted_bn <- bn.fit(pdag, data3)
print(fitted_bn)