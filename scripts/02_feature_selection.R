library(glmnet)
library(caret)
library(randomForest)
library(dplyr)
library(Boruta)

data <- read.csv("data/balanced_data_SMOTE.csv")
data$Diagnosis <- factor(data$Diagnosis)

set.seed(123)
data <- data[sample(nrow(data)), ]

set.seed(123)
train_index <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# LASSO ----
x <- model.matrix(Diagnosis ~ . - 1, data = train_data)
y <- train_data$Diagnosis

set.seed(123)
lasso_cv <- cv.glmnet(x, y, alpha = 1, family = "binomial")
best_lambda <- lasso_cv$lambda.min
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda, family = "binomial")

coefficients <- coef(lasso_model)[-1, , drop = FALSE]
lasso_importance <- data.frame(Feature = rownames(coefficients), Importance = abs(coefficients[, 1]))
lasso_importance <- lasso_importance %>% arrange(desc(Importance))

cat("LASSO Feature Importance:\n")
print(lasso_importance)

# RFE ----
rfe_control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfe_results <- rfe(
  x = train_data[, -ncol(train_data)],
  y = train_data$Diagnosis,
  sizes = c(5, 10, 15, 20),
  rfeControl = rfe_control
)

rfe_importance <- varImp(rfe_results)
rfe_importance <- data.frame(Feature = rownames(rfe_importance), Importance = rfe_importance$Overall)
rfe_importance <- rfe_importance %>% arrange(desc(Importance))

cat("RFE Feature Importance:\n")
print(rfe_importance)

# Boruta ----
set.seed(123)
boruta_result <- Boruta(Diagnosis ~ ., data = train_data, doTrace = 0)

boruta_importance <- attStats(boruta_result)
boruta_importance <- boruta_importance[order(-boruta_importance$meanImp), c("meanImp"), drop = FALSE]
boruta_importance$Feature <- rownames(boruta_importance)

cat("Boruta Feature Importance:\n")
print(boruta_importance)
