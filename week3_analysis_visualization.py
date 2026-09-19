import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load cleaned logistics dataset
df = pd.read_csv("cleaned_logistics_data.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())
# Step 2: Descriptive Statistics

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(df.describe())

print("\n========== CENTRAL TENDENCY ==========")

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

print("\nMean:")
print(df[numeric_columns].mean())

print("\nMedian:")
print(df[numeric_columns].median())

print("\nMode:")
print(df[numeric_columns].mode().iloc[0])
# Step 3: Delivery Status Analysis

print("\n========== DELIVERY STATUS ANALYSIS ==========")

status_counts = df["Delivery_Status"].value_counts()

print("\nDelivery Status Counts:")
print(status_counts)

# Bar chart
plt.figure(figsize=(8, 5))

status_counts.plot(kind="bar")

plt.title("On-Time vs Delayed Deliveries")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Deliveries")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("week3_delivery_status.png")

plt.show()

print("\nDelivery status visualization saved successfully!")
# Step 4: Delivery Time Distribution Analysis

print("\n========== DELIVERY TIME DISTRIBUTION ==========")

print("\nActual Delivery Time Statistics:")
print(df["Actual_Delivery_Time_hr"].describe())

# Histogram
plt.figure(figsize=(8, 5))

plt.hist(df["Actual_Delivery_Time_hr"], bins=15)

plt.title("Distribution of Actual Delivery Time")
plt.xlabel("Actual Delivery Time (Hours)")
plt.ylabel("Number of Deliveries")

plt.tight_layout()

plt.savefig("week3_delivery_time_distribution.png")

plt.show()

print("\nDelivery time distribution visualization saved successfully!")
# Step 5: Vehicle Type Analysis

print("\n========== VEHICLE TYPE ANALYSIS ==========")

vehicle_counts = df["Vehicle_Type"].value_counts()

print("\nVehicle Type Counts:")
print(vehicle_counts)

plt.figure(figsize=(8, 5))

vehicle_counts.plot(kind="bar")

plt.title("Deliveries by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Deliveries")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("week3_vehicle_type.png")

plt.show()

print("\nVehicle type visualization saved successfully!")
# Step 6: Traffic Level Analysis

print("\n========== TRAFFIC LEVEL ANALYSIS ==========")

traffic_counts = df["Traffic_Level"].value_counts()

print("\nTraffic Level Counts:")
print(traffic_counts)

plt.figure(figsize=(8, 5))

traffic_counts.plot(kind="bar")

plt.title("Deliveries by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Number of Deliveries")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("week3_traffic_level.png")

plt.show()

print("\nTraffic level visualization saved successfully!")
# Step 7: Distance vs Delivery Time Analysis

print("\n========== DISTANCE VS DELIVERY TIME ==========")

print("\nCorrelation between Distance and Actual Delivery Time:")

distance_time_corr = df["Distance_km"].corr(
    df["Actual_Delivery_Time_hr"]
)

print(distance_time_corr)

# Scatter plot
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Distance_km"],
    df["Actual_Delivery_Time_hr"]
)

plt.title("Distance vs Actual Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Actual Delivery Time (Hours)")

plt.tight_layout()

plt.savefig("week3_distance_vs_delivery_time.png")

plt.show()

print("\nDistance vs delivery time visualization saved successfully!")
# Step 8: Order Quantity vs Transportation Cost

print("\n========== ORDER QUANTITY VS TRANSPORTATION COST ==========")

print("\nCorrelation between Order Quantity and Transportation Cost:")

quantity_cost_corr = df["Order_Quantity"].corr(
    df["Transportation_Cost_INR"]
)

print(quantity_cost_corr)

# Scatter plot
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Order_Quantity"],
    df["Transportation_Cost_INR"]
)

plt.title("Order Quantity vs Transportation Cost")
plt.xlabel("Order Quantity")
plt.ylabel("Transportation Cost (INR)")

plt.tight_layout()

plt.savefig("week3_quantity_vs_cost.png")

plt.show()

print("\nOrder quantity vs transportation cost visualization saved successfully!")
# Step 9: Vehicle Utilization vs Delivery Time

print("\n========== VEHICLE UTILIZATION VS DELIVERY TIME ==========")

print("\nCorrelation between Vehicle Utilization and Actual Delivery Time:")

utilization_time_corr = df["Vehicle_Utilization_pct"].corr(
    df["Actual_Delivery_Time_hr"]
)

print(utilization_time_corr)

# Scatter plot
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Vehicle_Utilization_pct"],
    df["Actual_Delivery_Time_hr"]
)

plt.title("Vehicle Utilization vs Actual Delivery Time")
plt.xlabel("Vehicle Utilization (%)")
plt.ylabel("Actual Delivery Time (Hours)")

plt.tight_layout()

plt.savefig("week3_utilization_vs_delivery_time.png")

plt.show()

print("\nVehicle utilization vs delivery time visualization saved successfully!")
# Step 10: Correlation Heatmap

print("\n========== CORRELATION HEATMAP ==========")

correlation_data = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_data)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Logistics Variables")

plt.tight_layout()

plt.savefig("week3_correlation_heatmap.png")

plt.show()

print("\nCorrelation heatmap saved successfully!")
# Step 11: Logistics KPI Analysis

print("\n========== LOGISTICS KPI ANALYSIS ==========")

# On-Time Delivery Rate
on_time_count = (df["Delivery_Status"] == "On Time").sum()
total_deliveries = len(df)

on_time_rate = (on_time_count / total_deliveries) * 100

# Average Delivery Time
average_delivery_time = df["Actual_Delivery_Time_hr"].mean()

# Average Delay Time
average_delay_time = df["Delay_Time_hr"].mean()

# Average Transportation Cost
average_transportation_cost = df["Transportation_Cost_INR"].mean()

# Average Vehicle Utilization
average_vehicle_utilization = df["Vehicle_Utilization_pct"].mean()

print("\nKey Performance Indicators:")

print(f"On-Time Delivery Rate: {on_time_rate:.2f}%")
print(f"Average Delivery Time: {average_delivery_time:.2f} hours")
print(f"Average Delay Time: {average_delay_time:.2f} hours")
print(f"Average Transportation Cost: ₹{average_transportation_cost:.2f}")
print(f"Average Vehicle Utilization: {average_vehicle_utilization:.2f}%")

print("\nKPI analysis completed successfully!")
# Step 12: Key Insights and Recommendations

print("\n========== KEY INSIGHTS AND RECOMMENDATIONS ==========")

print("\nKEY INSIGHTS:")

print("1. The dataset contains 300 logistics delivery records.")

print("2. Delayed deliveries are higher than On-Time deliveries, "
      "with 181 delayed and 119 On-Time deliveries.")

print("3. Distance and Actual Delivery Time show a strong positive "
      "correlation of 0.775.")

print("4. Order Quantity and Transportation Cost show a weak positive "
      "correlation of 0.202.")

print("5. Vehicle Utilization and Actual Delivery Time show a correlation "
      "of approximately -0.034, indicating a very weak linear relationship.")

print("6. Medium traffic is the most common traffic level with 129 deliveries.")

print("7. Van and Mini Truck are the most frequently used vehicle types, "
      "with 79 deliveries each.")

print("\nRECOMMENDATIONS:")

print("1. Analyze long-distance routes to identify opportunities for "
      "reducing delivery time.")

print("2. Monitor delayed deliveries and investigate major causes of delay.")

print("3. Consider traffic conditions when planning delivery routes.")

print("4. Analyze transportation costs along with order quantity to "
      "identify cost optimization opportunities.")

print("5. Monitor vehicle utilization to support better vehicle planning.")

print("\nKey insights and recommendations completed successfully!")
# Step 13: Save Final Analysis Results

print("\n========== SAVING FINAL ANALYSIS RESULTS ==========")

# Create KPI summary
kpi_summary = pd.DataFrame({
    "KPI": [
        "On-Time Delivery Rate",
        "Average Delivery Time",
        "Average Delay Time",
        "Average Transportation Cost",
        "Average Vehicle Utilization"
    ],
    "Value": [
        on_time_rate,
        average_delivery_time,
        average_delay_time,
        average_transportation_cost,
        average_vehicle_utilization
    ]
})

# Save KPI results
kpi_summary.to_csv("week3_kpi_summary.csv", index=False)

# Save correlation matrix
correlation_data.to_csv("week3_correlation_matrix.csv")

print("\nKPI summary saved as: week3_kpi_summary.csv")
print("Correlation matrix saved as: week3_correlation_matrix.csv")

print("\nFinal analysis results saved successfully!")