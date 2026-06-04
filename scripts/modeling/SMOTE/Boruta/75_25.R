source("scripts/modeling/helpers.R")
data <- read.csv("data/balanced_data_SMOTE.csv")
set.seed(123); data <- data[sample(nrow(data)), ]
data$Diagnosis <- as.factor(data$Diagnosis)
set.seed(123); idx <- createDataPartition(data$Diagnosis, p = 0.75, list = FALSE)
print(run_all_models(data[idx, ], data[-idx, ]))
