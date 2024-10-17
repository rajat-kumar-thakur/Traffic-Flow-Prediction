library(readxl)
library(readr)
library(bnlearn)
library(Rgraphviz)

data_set <- read_csv("E:/College/3. Third Year/SEM V/2. Labs/CS367 Artificial Inteliigence/Traffic-Flow-Prediction/Data/TrafficDataSet.csv",  col_select = c("Time", "Dayoftheweek", "CarCount", "BikeCount", "BusCount", "TruckCount", "TrafficSituation"))

names(data_set)[names(data_set) == "Traffic Situation"] <- "TrafficSituation"

data1 = data_set
data2 = lapply(data1, as.factor)
data3 = data.frame(data2)

pdag = hc(data3)

traffic_graph = empty.graph(nodes(pdag))

for (node in nodes(pdag)) {
  if (node != "TrafficSituation") {
    traffic_graph = set.arc(traffic_graph, from = node, to = "TrafficSituation")
  }
}

par(mar = c(1, 1, 1, 1))
par(oma = c(0, 0, 2, 0))

png("traffic_situation_connections.png", width = 800, height = 600, res = 100)
plot(traffic_graph, main = "")
title("Connections to Traffic Situation", outer = TRUE)
dev.off()

plot(traffic_graph, main = "")
title("Custom Traffic Situation", outer = TRUE)