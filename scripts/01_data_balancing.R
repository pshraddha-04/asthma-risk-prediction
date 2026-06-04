f = read.csv("data/Asthma_Disease_Dataset.csv")

# Check for missing values
cat("Number of missing values in the dataset:\n")
print(sum(is.na(f)))

# Checking if data is balanced or not
cat("\nBefore Data balancing :\n")
data = table(f$Diagnosis)
print(data)
print(prop.table(data)*100)

# Drop non-relevant columns
f <- f[, !(names(f) %in% c("PatientID", "DoctorInCharge"))]

cat("--------------------------------------")

# Load initial libraries ----
library(ROSE)
library(smotefamily)

# Data balancing using --> ROSE ----
f$Diagnosis <- as.factor(f$Diagnosis)
f_balanced_rose <- ROSE(Diagnosis ~ ., data = f, seed = 123)$data

cat("\nAfter applying ROSE :\n")
print(table(f_balanced_rose$Diagnosis))
print(prop.table(table(f_balanced_rose$Diagnosis))*100)

cat("--------------------------------------")

# Data balancing using --> SMOTE ----
X <- f[, -ncol(f)]
y <- f$Diagnosis

smote_result <- SMOTE(X = X, target = y, K = 5, dup_size = 14)
f_balanced_smote <- smote_result$data
f_balanced_smote$Diagnosis <- as.factor(f_balanced_smote$class)

cat("\nAfter applying SMOTE :\n")
print(table(f_balanced_smote$Diagnosis))
print(prop.table(table(f_balanced_smote$Diagnosis)) * 100)

cat("--------------------------------------")

# Data balancing using --> Hybrid (Over+Under sampling) ----
f_balanced_hybrid <- ovun.sample(
  Diagnosis ~ ., 
  data = f, 
  method = "both", 
  p = 0.5, 
  N = sum(table(f$Diagnosis))
)$data

cat("\nAfter applying Hybrid Sampling (Over+Under) :\n")
print(table(f_balanced_hybrid$Diagnosis))
print(prop.table(table(f_balanced_hybrid$Diagnosis)) * 100)

# Save balanced data files ----
write.csv(f_balanced_rose, "data/balanced_data_ROSE.csv", row.names = FALSE)
write.csv(f_balanced_smote, "data/balanced_data_SMOTE.csv", row.names = FALSE)
write.csv(f_balanced_hybrid, "data/balanced_data_Hybrid.csv", row.names = FALSE)
