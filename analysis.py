import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans


# 1. Load dataset
df = pd.read_csv("logistics_delivery_dataset.csv")

print("First 5 rows:")
print(df.head())


# 2. Check shape
print("\nShape of dataset:")
print(df.shape)


# 3. Check columns
print("\nColumn names:")
print(df.columns)


# 4. Check missing values
print("\nMissing values:")
print(df.isnull().sum())


# 5. Check duplicates
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# 6. Remove duplicates
df.drop_duplicates(inplace=True)

print("\nDataset shape after removing duplicates:")
print(df.shape)


# 7. Check data types
print("\nData types:")
print(df.dtypes)


# 8. Convert Order_Date to date
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nOrder_Date converted successfully.")


# 9. First graph - On-Time vs Delayed Deliveries
status_counts = df["Delivery_Status"].value_counts()

status_counts.plot(kind="bar")

plt.title("On-Time vs Delayed Deliveries")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Deliveries")

plt.savefig("delivery_status.png")
plt.show()


# 10. Calculate On-Time Delivery Rate
total_deliveries = len(df)

on_time = len(df[df["Delivery_Status"] == "On Time"])

on_time_rate = (on_time / total_deliveries) * 100

print("\nOn-Time Delivery Rate:", on_time_rate, "%")


# 11. Calculate Average Delivery Time
average_delivery_time = df["Actual_Delivery_Time_hr"].mean()

print("Average Delivery Time:", average_delivery_time, "hours")


# 12. Calculate Transportation Cost Per Delivery
cost_per_delivery = df["Transportation_Cost_INR"].mean()

print("Transportation Cost Per Delivery:", cost_per_delivery, "INR")


# 13. Calculate Delay Time
df["Calculated_Delay_Time_hr"] = (
    df["Actual_Delivery_Time_hr"]
    - df["Expected_Delivery_Time_hr"]
)

print("\nDelay time calculated successfully.")


# 14. Extract day and month from Order_Date
df["Order_Day"] = df["Order_Date"].dt.day_name()

df["Order_Month"] = df["Order_Date"].dt.month

print("\nOrder day and month created.")


# 15. Linear Regression
X = df[["Distance_km", "Order_Quantity"]]

y = df["Actual_Delivery_Time_hr"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


predictions = model.predict(X_test)


# 16. Evaluate model
mae = mean_absolute_error(y_test, predictions)

print("\nLinear Regression Results:")
print("Mean Absolute Error:", mae)


# 17. K-Means Clustering
X_cluster = df[
    [
        "Distance_km",
        "Order_Quantity",
        "Transportation_Cost_INR"
    ]
]


kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)


df["Cluster"] = kmeans.fit_predict(X_cluster)


print("\nK-Means Clustering Results:")

print(
    df[
        [
            "Order_ID",
            "Distance_km",
            "Order_Quantity",
            "Transportation_Cost_INR",
            "Cluster"
        ]
    ].head(10)
)