# Logistics Delivery Performance Analysis and Route Optimization Using Python

## 📌 Project Overview

This project focuses on analyzing logistics and delivery operations using Python, data analysis, visualization, predictive modeling, and optimization techniques.

The project was completed progressively from **Week 1 to Week 4**, covering the complete workflow from understanding logistics data and preprocessing to advanced analysis, predictive modeling, and optimization recommendations.

The main objective is to identify delivery performance patterns, understand factors affecting delivery time and cost, and provide data-driven insights that can support better logistics planning and operational decision-making.

---

## 🎯 Project Objectives

- Analyze logistics and delivery performance.
- Clean and preprocess logistics data.
- Identify missing values, duplicates, invalid values, and outliers.
- Perform exploratory data analysis and visualization.
- Calculate important logistics KPIs.
- Analyze relationships between logistics variables.
- Build predictive models for delivery time.
- Evaluate predictive model performance.
- Identify important factors affecting delivery time.
- Provide optimization strategies for logistics operations.

---

## 📊 Dataset

The dataset contains **300 logistics delivery records** and **16 columns**.

### Dataset Features

| Feature | Description |
|---|---|
| Order_ID | Unique order identifier |
| Order_Date | Date of the order |
| Warehouse_Location | Warehouse location |
| Customer_Location | Customer location |
| Distance_km | Delivery distance in kilometers |
| Order_Quantity | Quantity of products ordered |
| Vehicle_Type | Type of vehicle used |
| Traffic_Level | Traffic condition |
| Expected_Delivery_Time_hr | Expected delivery time |
| Actual_Delivery_Time_hr | Actual delivery time |
| Fuel_Cost_INR | Fuel cost |
| Transportation_Cost_INR | Transportation cost |
| Delay_Time_hr | Delivery delay |
| Delivery_Status | Delivery status |
| Vehicle_Capacity | Vehicle capacity |
| Vehicle_Utilization_pct | Vehicle utilization percentage |

---

# 📅 Week 1 – Logistics Strategy, KPIs and Route Optimization

## Objective

The objective of Week 1 was to understand logistics operations, important logistics KPIs, delivery performance, and route optimization concepts.

## Activities Performed

- Studied logistics delivery processes.
- Analyzed important logistics KPIs.
- Explored delivery performance metrics.
- Studied factors affecting delivery time and transportation cost.
- Performed initial data analysis using Python.
- Created visualizations for logistics data.
- Explored route optimization concepts.
- Used Linear Regression and KMeans as part of the initial analysis workflow.

## Key Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Key Focus Areas

- Delivery performance
- Transportation cost
- Delivery time
- Vehicle utilization
- Route optimization
- Logistics KPIs

---

# 📅 Week 2 – Data Collection, Cleaning and Preprocessing

## Objective

The objective of Week 2 was to prepare the logistics dataset for further analysis by performing data cleaning and preprocessing.

## Activities Performed

### 1. Data Collection

The logistics dataset was loaded and examined using Python and Pandas.

### 2. Data Cleaning

The dataset was checked for:

- Missing values
- Duplicate records
- Invalid values
- Data inconsistencies

### 3. Outlier Analysis

Outliers were analyzed in numerical columns to identify unusual values that could affect the analysis.

### 4. Feature Engineering

Relevant features were prepared for further analysis and modeling.

### 5. Normalization

Numerical features were normalized to make them suitable for analytical and machine learning workflows.

### 6. Data Quality Verification

The final cleaned dataset was verified before moving to advanced analysis.

## Week 2 Outputs

- `cleaned_logistics_data.csv`
- `normalized_logistics_data.csv`
- `outliers_analysis.png`
- `week2_preprocessing.py`
- `Week2_Logistics_Preprocessing_Report.docx`

## Result

The cleaned dataset contained **300 records and 16 columns** and was prepared for Week 3 advanced analysis.

---

# 📅 Week 3 – Advanced Data Analysis and Visualization

## Objective

The objective of Week 3 was to perform advanced exploratory data analysis, identify relationships between logistics variables, calculate KPIs, and create meaningful visualizations.

## Activities Performed

- Analyzed delivery status.
- Analyzed delivery time distribution.
- Studied traffic-level patterns.
- Analyzed vehicle types.
- Examined distance versus delivery time.
- Examined order quantity versus transportation cost.
- Analyzed vehicle utilization versus delivery time.
- Created a correlation matrix and heatmap.
- Calculated important logistics KPIs.

## Key Results

### Delivery Status

- Delayed deliveries: **181**
- On-time deliveries: **119**
- On-Time Delivery Rate: **39.67%**

### Key KPIs

| KPI | Value |
|---|---:|
| On-Time Delivery Rate | 39.67% |
| Average Delivery Time | 5.99 hr |
| Average Delay | 1.74 hr |
| Average Transportation Cost | ₹1,284.00 |
| Average Vehicle Utilization | 48.10% |

### Correlation Analysis

- Distance vs Actual Delivery Time: **0.775**
- Order Quantity vs Transportation Cost: **0.202**
- Vehicle Utilization vs Actual Delivery Time: **-0.034**

The analysis showed a strong positive relationship between delivery distance and actual delivery time within this dataset.

## Visualizations Created

- Delivery status analysis
- Delivery time distribution
- Distance vs delivery time
- Quantity vs transportation cost
- Traffic level analysis
- Vehicle type analysis
- Vehicle utilization vs delivery time
- Correlation heatmap

## Week 3 Outputs

- `week3_analysis_visualization.py`
- `week3_correlation_heatmap.png`
- `week3_correlation_matrix.csv`
- `week3_delivery_status.png`
- `week3_delivery_time_distribution.png`
- `week3_distance_vs_delivery_time.png`
- `week3_kpi_summary.csv`
- `week3_quantity_vs_cost.png`
- `week3_traffic_level.png`
- `week3_utilization_vs_delivery_time.png`
- `week3_vehicle_type.png`
- `Week3_Logistics_Analysis_Report.docx`

---

# 📅 Week 4 – Predictive Modeling and Optimization

## Objective

The objective of Week 4 was to apply predictive modeling techniques to forecast logistics delivery time and develop optimization recommendations.

## Target Variable

The target variable selected for prediction was:

`Actual_Delivery_Time_hr`

## Input Features

The following features were used:

- Distance_km
- Order_Quantity
- Expected_Delivery_Time_hr
- Fuel_Cost_INR
- Transportation_Cost_INR
- Delay_Time_hr
- Vehicle_Capacity
- Vehicle_Utilization_pct

## Models Implemented

### 1. Linear Regression

Linear Regression was implemented to predict actual delivery time.

### 2. Decision Tree Regression

Decision Tree Regression was implemented to capture non-linear relationships between logistics features and delivery time.

## Model Evaluation

The models were evaluated using:

- MAE – Mean Absolute Error
- RMSE – Root Mean Squared Error
- R² – R-squared

### Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ~0.0000 | ~0.0000 | 1.0000 |
| Decision Tree Regression | 0.5075 | 0.6277 | 0.9272 |

## Feature Importance

Decision Tree feature importance showed:

| Feature | Importance |
|---|---:|
| Distance_km | 0.499963 |
| Delay_Time_hr | 0.354942 |
| Expected_Delivery_Time_hr | 0.145096 |
| Order_Quantity | 0 |
| Fuel_Cost_INR | 0 |
| Transportation_Cost_INR | 0 |
| Vehicle_Capacity | 0 |
| Vehicle_Utilization_pct | 0 |

## Predictive Insights

The analysis identified:

1. Distance was the highest-ranked feature in the Decision Tree feature importance results.
2. Delay time was the second-highest feature.
3. Expected delivery time was the third-highest feature.
4. Predictive modeling can support delivery planning and delay monitoring.
5. The results can be used as a starting point for logistics optimization.

## Optimization Recommendations

### Route Optimization

Long-distance routes can be monitored and analyzed carefully because distance had the highest feature importance in the Decision Tree analysis.

### Delay Management

Delay time should be monitored to identify recurring causes of delivery delays.

### Delivery Planning

Expected delivery time and route information can be considered when planning deliveries.

### Vehicle Planning

Vehicle capacity and utilization information can be considered during vehicle assignment.

### Cost Monitoring

Fuel and transportation costs can be monitored together with delivery performance.

## Week 4 Outputs

- `week4_predictive_modeling.py`
- `week4_model_performance.csv`
- `week4_prediction_results.csv`
- `week4_feature_importance.csv`
- `week4_actual_vs_predicted.png`
- `week4_feature_importance.png`
- `Week4_Logistics_Predictive_Modeling_Report.docx`

---

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Linear Regression**
- **Decision Tree Regression**
- **KMeans Clustering**
- **Git**
- **GitHub**

---

# 📈 Overall Project Workflow

```text
Raw Logistics Dataset
        ↓
Data Understanding
        ↓
Week 1
Logistics Strategy + KPIs + Route Optimization
        ↓
Week 2
Data Cleaning + Preprocessing + Outlier Analysis
        ↓
Week 3
Advanced Data Analysis + Visualization + KPI Analysis
        ↓
Week 4
Predictive Modeling + Model Evaluation
        ↓
Feature Importance Analysis
        ↓
Optimization Recommendations
        ↓
Data-Driven Logistics Insights
Logistics-Delivery-Analysis/
│
├── analysis.py
├── logistics_delivery_dataset.csv
├── cleaned_logistics_data.csv
├── normalized_logistics_data.csv
│
├── week2_preprocessing.py
├── outliers_analysis.png
├── Week2_Logistics_Preprocessing_Report.docx
│
├── week3_analysis_visualization.py
├── week3_correlation_heatmap.png
├── week3_correlation_matrix.csv
├── week3_delivery_status.png
├── week3_delivery_time_distribution.png
├── week3_distance_vs_delivery_time.png
├── week3_kpi_summary.csv
├── week3_quantity_vs_cost.png
├── week3_traffic_level.png
├── week3_utilization_vs_delivery_time.png
├── week3_vehicle_type.png
├── Week3_Logistics_Analysis_Report.docx
│
├── week4_predictive_modeling.py
├── week4_model_performance.csv
├── week4_prediction_results.csv
├── week4_feature_importance.csv
├── week4_actual_vs_predicted.png
├── week4_feature_importance.png
├── Week4_Logistics_Predictive_Modeling_Report.docx
│
├── Week1_Logistics_Project_Report.docx
└── README.md
🔍 Key Project Findings

The project provided insights into different aspects of logistics operations:

Delivery status and on-time delivery performance were analyzed.
Average delivery time was approximately 5.99 hours.
Average delivery delay was approximately 1.74 hours.
Average transportation cost was approximately ₹1,284.
Average vehicle utilization was approximately 48.10%.
Distance showed a correlation of approximately 0.775 with actual delivery time in the analyzed dataset.
Predictive models were developed to estimate actual delivery time.
Decision Tree feature importance highlighted distance, delay time, and expected delivery time as important features.
⚠️ Model Limitation

The Linear Regression model produced an almost perfect test result with an R² of 1.0000. This result should be interpreted carefully because some input variables, particularly expected delivery time and delay time, may have a direct mathematical relationship with actual delivery time in this dataset.

Therefore, the model results demonstrate the analytical workflow but should not be treated as proof of real-world predictive performance without testing on more independent data and reviewing potential target leakage.

🚀 Future Scope

Future improvements can include:

Using larger real-world logistics datasets.
Adding GPS and geographical information.
Implementing real-time traffic data.
Using advanced machine learning models.
Applying cross-validation and hyperparameter tuning.
Developing real-time delivery prediction.
Implementing optimization algorithms such as route optimization.
Building an interactive dashboard using Power BI or a web application.
Monitoring logistics KPIs in real time.
👩‍💻 Project Developed By

Priya Bapuso Kale

Master of Computer Applications
G. H. Raisoni College of Engineering and Management, Pune

📌 Conclusion

This four-week project covered the complete data analytics and machine learning workflow for logistics delivery analysis. Starting from logistics concepts and KPI analysis, the project progressed through data cleaning, preprocessing, exploratory analysis, visualization, predictive modeling, feature importance analysis, and optimization recommendations.

The project demonstrates practical experience in Python, Data Analysis, Data Visualization, Machine Learning, Predictive Modeling, and GitHub-based project management.