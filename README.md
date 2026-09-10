# Logistics Delivery Performance Analysis and Route Optimization Using Python

## Project Overview

This project focuses on analyzing logistics delivery data to understand delivery performance, identify delays, calculate important logistics KPIs, and explore methods for improving delivery efficiency.

Python and data science techniques are used to perform data cleaning, exploratory data analysis, KPI calculation, delivery-time prediction, and customer/order clustering.

## Objectives

- Analyze logistics delivery performance.
- Calculate important logistics KPIs.
- Identify delayed and on-time deliveries.
- Analyze delivery time and transportation costs.
- Predict delivery time using Linear Regression.
- Group similar delivery orders using K-Means Clustering.
- Understand the concept of route optimization.
- Identify opportunities to improve logistics efficiency.

## Dataset

The project uses a logistics delivery dataset containing **300 delivery records and 16 columns**.

Important columns include:

- Order_ID
- Order_Date
- Warehouse_Location
- Customer_Location
- Distance_km
- Order_Quantity
- Vehicle_Type
- Traffic_Level
- Expected_Delivery_Time_hr
- Actual_Delivery_Time_hr
- Fuel_Cost_INR
- Transportation_Cost_INR
- Delay_Time_hr
- Delivery_Status
- Vehicle_Capacity
- Vehicle_Utilization_pct

## Key Performance Indicators (KPIs)

The following KPIs are analyzed:

1. On-Time Delivery Rate
2. Average Delivery Time
3. Transportation Cost Per Delivery
4. Average Delay Time
5. Vehicle Utilization

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- VS Code
- CSV Dataset

## Data Analysis

The project performs:

- Dataset loading
- Data cleaning
- Missing-value checking
- Duplicate checking
- Data type conversion
- Delivery status analysis
- KPI calculation
- Delay analysis
- Date-based analysis
- Data visualization

## Machine Learning Techniques

### Linear Regression

Linear Regression is used to analyze the relationship between delivery time and factors such as:

- Distance
- Order Quantity

The model is evaluated using Mean Absolute Error (MAE).

### K-Means Clustering

K-Means Clustering is used to group delivery orders based on:

- Distance
- Order Quantity
- Transportation Cost

This helps identify different types of delivery patterns.

## Route Optimization

Route optimization aims to find an efficient sequence for delivering orders while minimizing:

- Total distance
- Delivery time
- Transportation cost

In a real-world implementation, algorithms and libraries such as Google OR-Tools can be used for vehicle routing and capacity constraints.

Author

Priya Kale

MCA Graduate
