library(randomForest)
library(xgboost)
library(caret)
library(e1071)
library(nnet)

feature_sets <- list(
  All  = NULL,  # filled at runtime
  Top8 = c("Age", "BMI", "NighttimeSymptoms", "ChestTightness", "DietQuality", "Wheezing", "ExerciseInduced", "HistoryOfAllergies"),
  Top9 = c("Age", "BMI", "NighttimeSymptoms", "ChestTightness", "DietQuality", "Wheezing", "ExerciseInduced", "HistoryOfAllergies", "DustExposure"),
  Top10 = c("Age", "BMI", "NighttimeSymptoms", "ChestTightness", "DietQuality", "Wheezing", "ExerciseInduced", "HistoryOfAllergies", "DustExposure", "FamilyHistoryAsthma")
)

train_and_evaluate <- function(train, test, features, model_name) {
  train_subset <- train[, c(features, "Diagnosis")]
  test_subset  <- test[, c(features, "Diagnosis")]

  if (model_name == "Random Forest") {
    model <- randomForest(Diagnosis ~ ., data = train_subset, ntree = 150, mtry = 2, maxnodes = 10)

  } else if (model_name == "SVM") {
    model <- svm(Diagnosis ~ ., data = train_subset, kernel = "radial")

  } else if (model_name == "XGBoost") {
    train_matrix <- model.matrix(Diagnosis ~ . -1, data = train_subset)
    test_matrix  <- model.matrix(Diagnosis ~ . -1, data = test_subset)
    train_label  <- as.numeric(train_subset$Diagnosis) - 1
    test_label   <- as.numeric(test_subset$Diagnosis) - 1
    xgb_train <- xgb.DMatrix(data = train_matrix, label = train_label)
    xgb_test  <- xgb.DMatrix(data = test_matrix,  label = test_label)
    model <- xgboost(data = xgb_train, nrounds = 100, reg_alpha = 30, max_depth = 4,
                     min_child_weight = 50, gamma = 5, subsample = 0.5,
                     colsample_bytree = 0.5, learning_rate = 0.005, reg_lambda = 10,
                     scale_pos_weight = 2, objective = "binary:logistic",
                     eval_metric = "logloss", early_stopping_rounds = 10, verbose = 0)

  } else if (model_name == "Neural Network") {
    normalize <- function(x) (x - min(x)) / (max(x) - min(x))
    train_subset[, features] <- lapply(train_subset[, features], normalize)
    test_subset[, features]  <- lapply(test_subset[, features], normalize)
    model <- nnet(Diagnosis ~ ., data = train_subset, size = 20, decay = 0.01, maxit = 500, trace = FALSE)
  }

  if (model_name == "XGBoost") {
    pred <- ifelse(predict(model, xgb_test) > 0.5, 1, 0)
  } else if (model_name == "Neural Network") {
    pred <- ifelse(predict(model, test_subset, type = "raw") > 0.5, 1, 0)
  } else {
    pred <- predict(model, test_subset)
  }

  cm <- confusionMatrix(factor(pred, levels = c(0, 1)), test_subset$Diagnosis)
  list(Model = model_name,
       Accuracy    = cm$overall["Accuracy"],
       Sensitivity = cm$byClass["Sensitivity"],
       Specificity = cm$byClass["Specificity"],
       Precision   = cm$byClass["Precision"],
       F1_Score    = 2 * (cm$byClass["Precision"] * cm$byClass["Sensitivity"]) /
                         (cm$byClass["Precision"] + cm$byClass["Sensitivity"]))
}

run_all_models <- function(train_data, test_data) {
  feature_sets$All <- names(train_data)[names(train_data) != "Diagnosis"]
  results <- list()
  for (fs_name in names(feature_sets)) {
    for (model in c("Random Forest", "SVM", "XGBoost", "Neural Network")) {
      results[[paste(model, fs_name, sep = "_")]] <-
        train_and_evaluate(train_data, test_data, feature_sets[[fs_name]], model)
    }
  }
  do.call(rbind, lapply(results, as.data.frame))
}
