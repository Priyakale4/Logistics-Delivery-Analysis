import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# Step 1: Load the logistics dataset

df = pd.read_csv("logistics_delivery_dataset.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)
# Step 2: Understand the dataset characteristics

print("\nDataset Information:")
df.info()

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)
# Step 3: Check for missing values

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())
# Step 4: Handle missing values

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )

print("\nMissing values have been handled successfully.")

print("\nMissing Values After Handling:")
print(df.isnull().sum().sum())
# Step 5: Check for duplicate records

print("\nNumber of Duplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nDuplicate records removed successfully.")

print("\nDataset shape after removing duplicates:")
print(df.shape)
# Step 6: Check for invalid values

print("\nInvalid Value Check:")

print("Negative Distance values:",
      (df["Distance_km"] < 0).sum())

print("Negative Order Quantity values:",
      (df["Order_Quantity"] < 0).sum())

print("Invalid Expected Delivery Time values:",
      (df["Expected_Delivery_Time_hr"] <= 0).sum())

print("Invalid Actual Delivery Time values:",
      (df["Actual_Delivery_Time_hr"] <= 0).sum())

print("Negative Fuel Cost values:",
      (df["Fuel_Cost_INR"] < 0).sum())

print("Negative Transportation Cost values:",
      (df["Transportation_Cost_INR"] < 0).sum())

print("Invalid Vehicle Capacity values:",
      (df["Vehicle_Capacity"] <= 0).sum())

print("Vehicle Utilization below 0%:",
      (df["Vehicle_Utilization_pct"] < 0).sum())

print("Vehicle Utilization above 100%:",
      (df["Vehicle_Utilization_pct"] > 100).sum())
# Step 7: Detect outliers using IQR method

print("\nOutlier Detection using IQR:")

numeric_columns = [
    "Distance_km",
    "Order_Quantity",
    "Expected_Delivery_Time_hr",
    "Actual_Delivery_Time_hr",
    "Fuel_Cost_INR",
    "Transportation_Cost_INR",
    "Delay_Time_hr",
    "Vehicle_Capacity",
    "Vehicle_Utilization_pct"
]

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"{column}: {len(outliers)} outliers")
    # Step 8: Visualize outliers using boxplots

plt.figure(figsize=(12, 6))

df[numeric_columns].boxplot()

plt.title("Boxplot of Numerical Logistics Variables")
plt.xlabel("Numerical Variables")
plt.ylabel("Values")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("outliers_analysis.png")

plt.show()

print("\nOutlier visualization created successfully.")
# Step 9: Handle outliers using IQR capping

print("\nHandling Outliers using IQR Capping:")

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Cap values below lower limit
    df[column] = df[column].clip(lower=lower_limit)

    # Cap values above upper limit
    df[column] = df[column].clip(upper=upper_limit)

    print(f"{column}: Outliers capped successfully.")

print("\nOutlier handling completed successfully.")
# Step 10: Save the cleaned dataset

df.to_csv("cleaned_logistics_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("File name: cleaned_logistics_data.csv")

print("\nCleaned Dataset Shape:")
print(df.shape)
# Step 11: Feature Engineering

print("\nCreating new features...")

# Feature 1: Delivery Delay Status
df["Delivery_Delay_Status"] = np.where(
    df["Delay_Time_hr"] > 0,
    "Delayed",
    "On Time"
)

# Feature 2: Transportation Cost per Quantity
df["Cost_Per_Quantity"] = (
    df["Transportation_Cost_INR"] / df["Order_Quantity"]
)

# Feature 3: Difference between actual and expected delivery time
df["Delivery_Time_Difference_hr"] = (
    df["Actual_Delivery_Time_hr"]
    - df["Expected_Delivery_Time_hr"]
)

print("\nNew features created successfully!")

print("\nNew Feature Preview:")
print(
    df[
        [
            "Order_ID",
            "Delivery_Delay_Status",
            "Cost_Per_Quantity",
            "Delivery_Time_Difference_hr"
        ]
    ].head()
)
# Step 12: Normalize numerical data using Min-Max Scaling

print("\nStarting Min-Max Normalization...")

normalization_columns = [
    "Distance_km",
    "Order_Quantity",
    "Expected_Delivery_Time_hr",
    "Actual_Delivery_Time_hr",
    "Fuel_Cost_INR",
    "Transportation_Cost_INR",
    "Delay_Time_hr",
    "Vehicle_Capacity",
    "Vehicle_Utilization_pct",
    "Cost_Per_Quantity",
    "Delivery_Time_Difference_hr"
]

scaler = MinMaxScaler()

df[normalization_columns] = scaler.fit_transform(
    df[normalization_columns]
)

print("\nNormalization completed successfully!")

print("\nNormalized Data Preview:")
print(df[normalization_columns].head())
# Step 13: Save the normalized dataset

df.to_csv("normalized_logistics_data.csv", index=False)

print("\nNormalized dataset saved successfully!")
print("File name: normalized_logistics_data.csv")

print("\nFinal Dataset Shape:")
print(df.shape)
# Step 14: Final data quality verification

print("\n========== FINAL DATA QUALITY CHECK ==========")

# Check missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check duplicate records
print("\nTotal Duplicate Records:")
print(df.duplicated().sum())

# Check dataset shape
print("\nFinal Dataset Shape:")
print(df.shape)

# Check data types
print("\nFinal Data Types:")
print(df.dtypes)

print("\n========== DATA PREPROCESSING COMPLETED ==========")
# Step 15: Final preprocessing summary

print("\n========== PREPROCESSING SUMMARY ==========")

print("Original Dataset File: logistics_delivery_dataset.csv")
print("Cleaned Dataset File: cleaned_logistics_data.csv")
print("Normalized Dataset File: normalized_logistics_data.csv")

print("\nFinal Number of Records:", len(df))
print("Final Number of Columns:", len(df.columns))

print("\nTotal Missing Values:", df.isnull().sum().sum())
print("Total Duplicate Records:", df.duplicated().sum())

print("\nPreprocessing steps completed:")
print("1. Data collection and loading")
print("2. Dataset structure analysis")
print("3. Missing value detection and handling")
print("4. Duplicate detection and removal")
print("5. Invalid value detection")
print("6. Outlier detection")
print("7. Outlier handling using IQR capping")
print("8. Feature engineering")
print("9. Min-Max normalization")
print("10. Final data quality verification")

print("\n========== PREPROCESSING COMPLETED SUCCESSFULLY ==========")