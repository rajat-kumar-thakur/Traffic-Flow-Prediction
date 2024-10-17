library(readxl)
library(readr)
library(bnlearn)
library(Rgraphviz)
library(caret)

data_set <- read_csv("E:/College/3. Third Year/SEM V/2. Labs/CS367 Artificial Inteliigence/Traffic-Flow-Prediction/Data/TrafficDataSet.csv",
                     col_select = c("Time", "Dayoftheweek", "CarCount", "BikeCount", "BusCount", "TruckCount", "TrafficSituation"))

names(data_set)[names(data_set) == "Traffic Situation"] <- "TrafficSituation"

data_set <- lapply(data_set, as.factor)
data_set <- data.frame(data_set)

ensure_two_levels <- function(data) {
  for (col in names(data)) {
    if (is.factor(data[[col]]) && nlevels(data[[col]]) < 2) {
      levels(data[[col]]) <- c(levels(data[[col]]), paste0(levels(data[[col]])[1], "_dummy"))
    }
  }
  return(data)
}

data_set <- ensure_two_levels(data_set)

set.seed(123)
train_index <- createDataPartition(data_set$TrafficSituation, p = 0.8, list = FALSE)
train_data <- data_set[train_index, ]
test_data <- data_set[-train_index, ]

train_data <- ensure_two_levels(train_data)
test_data <- ensure_two_levels(test_data)

bn_structure <- hc(train_data)

bn_fit <- bn.fit(bn_structure, train_data)

predict_traffic <- function(model, new_data) {
  predictions <- predict(model, new_data, node = "TrafficSituation")
  return(predictions)
}

predictions <- predict_traffic(bn_fit, test_data)

accuracy <- sum(predictions == test_data$TrafficSituation) / nrow(test_data)

cat("Prediction accuracy:", round(accuracy * 100, 2), "%\n")

predict_for_day_time <- function(model, day, time, car_count, bike_count, bus_count, truck_count) {
  new_data <- data.frame(
    Dayoftheweek = factor(day, levels = levels(train_data$Dayoftheweek)),
    Time = factor(time, levels = levels(train_data$Time)),
    CarCount = factor(car_count, levels = levels(train_data$CarCount)),
    BikeCount = factor(bike_count, levels = levels(train_data$BikeCount)),
    BusCount = factor(bus_count, levels = levels(train_data$BusCount)),
    TruckCount = factor(truck_count, levels = levels(train_data$TruckCount))
  )

  prediction <- predict_traffic(model, new_data)
  return(prediction)
}

example_prediction <- predict_for_day_time(bn_fit, "Monday", "00:00", "50", "10", "5", "2")
cat("Predicted traffic situation for Monday at 08:00:", as.character(example_prediction), "\n")

png("traffic_situation_network.png", width = 800, height = 600, res = 100)
plot(bn_structure, main = "Traffic Situation Bayesian Network")
dev.off()
