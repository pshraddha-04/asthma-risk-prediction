f = read.csv("Asthma_Disease_Dataset.csv")

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

# Convert target variable to a factor(categorical variable)
f$Diagnosis <- as.factor(f$Diagnosis)
f_balanced_rose <- ROSE(Diagnosis ~ ., data = f, seed = 123)$data

# Check the new class distribution
cat("\nAfter applying ROSE :\n")
print(table(f_balanced_rose$Diagnosis))
print(prop.table(table(f_balanced_rose$Diagnosis))*100)

cat("--------------------------------------")

# Data balancing using --> SMOTE ----

# Separate features (X) and target (y)
X <- f[, -ncol(f)]   # All columns except 'Diagnosis'
y <- f$Diagnosis     # Target variable

# Apply SMOTE 
smote_result <- SMOTE(X = X, target = y, K = 5, dup_size = 14)

# Extract balanced dataset
f_balanced_smote <- smote_result$data

# renamed target variable class back to Diagnosis 
f_balanced_smote$Diagnosis <- as.factor(f_balanced_smote$class)

# Check new class distribution
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

# Check new class distribution
cat("\nAfter applying Hybrid Sampling (Over+Under) :\n")
print(table(f_balanced_hybrid$Diagnosis))
print(prop.table(table(f_balanced_hybrid$Diagnosis)) * 100)


#Balanced data files ----

# write.csv(f_balanced_rose, "balanced_data_ROSE.csv", row.names = FALSE)
# write.csv(f_balanced_smote, "balanced_data_SMOTE.csv", row.names = FALSE)
# write.csv(f_balanced_hybrid, "balanced_data_Hybrid.csv", row.names = FALSE)
 



