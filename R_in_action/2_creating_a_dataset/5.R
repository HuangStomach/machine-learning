patient_id <- c(1, 2, 3, 4)
age <- c(25, 34, 28, 52)
diabetes <- c("Type1", "Type2", "Type1", "Type1")
status <- c("Poor", "Improved", "Excellent", "Poor")
patient_data <- data.frame(patient_id, age, diabetes, status)

print(patient_data[1:2])
print(patient_data[c("diabetes", "status")])
print(patient_data$age)
