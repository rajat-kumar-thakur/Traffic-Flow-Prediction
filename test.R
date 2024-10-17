# Install and load necessary packages
if (!require(rstanarm)) install.packages("rstanarm")
library(rstanarm)

# Generate sample data
set.seed(123)
n <- 1000
time_of_day <- sample(0:23, n, replace = TRUE)
day_of_week <- sample(1:7, n, replace = TRUE)
traffic_flow <- rnorm(n, mean = 100 + 5 * sin(time_of_day * pi/12) + 10 * (day_of_week <= 5), sd = 20)

# Create data frame
data <- data.frame(
  time_of_day = time_of_day,
  day_of_week = day_of_week,
  traffic_flow = traffic_flow
)

# Fit the Bayesian regression model
model <- stan_glm(
  traffic_flow ~ sin(time_of_day * pi/12) + I(day_of_week <= 5),
  data = data,
  family = gaussian(),
  prior = normal(location = c(100, 0, 0), scale = c(10, 5, 5)),
  prior_intercept = normal(100, 10),
  prior_aux = exponential(1),
  chains = 4,
  iter = 2000
)

# Summarize results
print(summary(model))

# Function to predict traffic flow
predict_traffic_flow <- function(time, day, model) {
  new_data <- data.frame(
    time_of_day = time,
    day_of_week = day
  )

  predictions <- posterior_predict(model, newdata = new_data)

  mean_prediction <- mean(predictions)
  ci <- quantile(predictions, c(0.025, 0.975))

  return(list(mean = mean_prediction, ci_lower = ci[1], ci_upper = ci[2]))
}

# Example prediction
new_time <- 14  # 2 PM
new_day <- 3    # Wednesday
prediction <- predict_traffic_flow(new_time, new_day, model)

print(paste("Predicted traffic flow at 2 PM on Wednesday:", round(prediction$mean, 2)))
print(paste("95% CI: [", round(prediction$ci_lower, 2), ",", round(prediction$ci_upper, 2), "]"))

# Plot posterior distributions of coefficients
plot(model, plotfun = "areas", prob = 0.95)