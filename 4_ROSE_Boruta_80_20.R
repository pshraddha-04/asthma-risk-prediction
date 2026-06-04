# Load necessary libraries
library(randomForest)  # For Random Forest
library(xgboost)       # For XGBoost
library(caret)         # For data splitting and evaluation metrics
library(e1071)         # For SVM
library(nnet)          # For Neural Network

# Load the dataset
data <- read.csv("balanced_data_ROSE.csv")

# Shuffle the data
set.seed(123)  # For reproducibility
data <- data[sample(nrow(data)), ]

# Split data into 80% training and 20% testing
set.seed(123)
train_index <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Convert target variable 'diagnosis' to a factor
train_data$Diagnosis <- as.factor(train_data$Diagnosis)
test_data$Diagnosis <- as.factor(test_data$Diagnosis)

# Define feature sets
features_all <- names(train_data)[names(train_data) != "Diagnosis"]
features_top8 <- c("Age", "BMI", "NighttimeSymptoms", "ChestTightness", "DietQuality", "Wheezing", "ExerciseInduced", "HistoryOfAllergies")
features_top9 <- c(features_top8, "DustExposure")
features_top10 <- c(features_top9, "FamilyHistoryAsthma")

feature_sets <- list(All = features_all, Top8 = features_top8, Top9 = features_top9, Top10 = features_top10)

# Function to train and evaluate models
train_and_evaluate <- function(train, test, features, model_name) {
  train_subset <- train[, c(features, "Diagnosis")]
  test_subset <- test[, c(features, "Diagnosis")]
  
  # Train Model
  if (model_name == "Random Forest") {
    model <- randomForest(
      Diagnosis ~ ., 
      data = train_subset, 
      ntree = 150, 
      mtry = 2, 
      maxnodes = 10)
  } 
  else if (model_name == "SVM") {
    model <- svm(
      Diagnosis ~ ., 
      data = train_subset, 
      kernel = "radial")
  } 
  else if (model_name == "XGBoost") {
    train_matrix <- model.matrix(Diagnosis ~ . -1, data = train_subset)
    test_matrix <- model.matrix(Diagnosis ~ . -1, data = test_subset)
    
    train_label <- as.numeric(train_subset$Diagnosis) - 1
    test_label <- as.numeric(test_subset$Diagnosis) - 1
    
    xgb_train <- xgb.DMatrix(data = train_matrix, label = train_label)
    xgb_test <- xgb.DMatrix(data = test_matrix, label = test_label)
    
    model <- xgboost(
      data = xgb_train, 
      nrounds = 100,
      reg_alpha = 30, 
      max_depth = 4,  
      min_child_weight = 50,  
      gamma = 5,  # Introduce a penalty for tree splits  
      subsample = 0.5,  
      colsample_bytree = 0.5,  
      learning_rate = 0.005,  
      reg_lambda = 10,  
      scale_pos_weight = 2,  # Try 1.5 or 2
      objective = "binary:logistic",
      eval_metric = "logloss",
      early_stopping_rounds = 10,
      verbose = 0)
  } 
  else if (model_name == "Neural Network") {
    normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
    
    train_subset[, features] <- lapply(train_subset[, features], normalize)
    test_subset[, features] <- lapply(test_subset[, features], normalize)
    
    model <- nnet(
      Diagnosis ~ ., 
      data = train_subset, 
      size = 20, 
      decay = 0.01, 
      maxit = 500, 
      trace = FALSE)  
  }
  
  # Predictions
  if (model_name == "XGBoost") {
    pred_prob <- predict(model, xgb_test)
    pred <- ifelse(pred_prob > 0.5, 1, 0)
  } 
  else if (model_name == "Neural Network") {
    pred_prob <- predict(model, test_subset, type = "raw")
    pred <- ifelse(pred_prob > 0.5, 1, 0)
  } 
  else {
    pred <- predict(model, test_subset)
  }
  
  # Evaluation
  cm <- confusionMatrix(factor(pred, levels = c(0,1)), test_subset$Diagnosis)
  return(list(Model = model_name, Accuracy = cm$overall["Accuracy"], Sensitivity = cm$byClass["Sensitivity"], Specificity = cm$byClass["Specificity"], Precision = cm$byClass["Precision"], F1_Score = 2 * (cm$byClass["Precision"] * cm$byClass["Sensitivity"]) / (cm$byClass["Precision"] + cm$byClass["Sensitivity"])))
}

# Run models for each feature set
results <- list()
models <- c("Random Forest", "SVM", "XGBoost", "Neural Network")
for (features in names(feature_sets)) {
  for (model in models) {
    results[[paste(model, features, sep = "_")]] <- train_and_evaluate(train_data, test_data, feature_sets[[features]], model)
  }
}

# Convert results to a data frame and print
results_df <- do.call(rbind, lapply(results, as.data.frame))
print(results_df)
