import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load cleaned logistics dataset
df = pd.read_csv("cleaned_logistics_data.csv")

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())
# Step 3: Define Prediction Problem

print("\n========== PREDICTION PROBLEM ==========")

# Target variable
target = "Actual_Delivery_Time_hr"

# Input features
features = [
    "Distance_km",
    "Order_Quantity",
    "Expected_Delivery_Time_hr",
    "Fuel_Cost_INR",
    "Transportation_Cost_INR",
    "Delay_Time_hr",
    "Vehicle_Capacity",
    "Vehicle_Utilization_pct"
]

X = df[features]
y = df[target]

print("\nTarget Variable:")
print(target)

print("\nInput Features:")
print(features)

print("\nFeature Data Shape:")
print(X.shape)

print("\nTarget Data Shape:")
print(y.shape)
# Step 4: Train-Test Split

from sklearn.model_selection import train_test_split

print("\n========== TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
# Step 5: Linear Regression Model

from sklearn.linear_model import LinearRegression

print("\n========== LINEAR REGRESSION MODEL ==========")

# Create model
linear_model = LinearRegression()

# Train model
linear_model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# Make predictions
y_pred_linear = linear_model.predict(X_test)

print("\nFirst 10 Actual Values:")
print(y_test.head(10).values)

print("\nFirst 10 Predicted Values:")
print(y_pred_linear[:10])
# Step 6: Evaluate Linear Regression Model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n========== LINEAR REGRESSION EVALUATION ==========")

mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print(f"\nMean Absolute Error (MAE): {mae_linear:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_linear:.4f}")
print(f"R-squared (R²): {r2_linear:.4f}")
# Step 7: Decision Tree Regression Model

from sklearn.tree import DecisionTreeRegressor

print("\n========== DECISION TREE REGRESSION ==========")

# Create model
tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

# Train model
tree_model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

# Make predictions
y_pred_tree = tree_model.predict(X_test)

print("\nFirst 10 Decision Tree Predictions:")
print(y_pred_tree[:10])
# Step 8: Evaluate Decision Tree Model

print("\n========== DECISION TREE EVALUATION ==========")

mae_tree = mean_absolute_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)

print(f"\nMean Absolute Error (MAE): {mae_tree:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_tree:.4f}")
print(f"R-squared (R²): {r2_tree:.4f}")
# Step 9: Model Performance Comparison

print("\n========== MODEL PERFORMANCE COMPARISON ==========")

model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree Regression"
    ],
    "MAE": [
        mae_linear,
        mae_tree
    ],
    "RMSE": [
        rmse_linear,
        rmse_tree
    ],
    "R2_Score": [
        r2_linear,
        r2_tree
    ]
})

print("\nModel Performance:")
print(model_results)

# Save model performance results
model_results.to_csv("week4_model_performance.csv", index=False)

print("\nModel performance saved as: week4_model_performance.csv")
# Step 10: Actual vs Predicted Delivery Time

print("\n========== ACTUAL VS PREDICTED DELIVERY TIME ==========")

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred_linear)

plt.xlabel("Actual Delivery Time (Hours)")
plt.ylabel("Predicted Delivery Time (Hours)")
plt.title("Actual vs Predicted Delivery Time - Linear Regression")

plt.tight_layout()

plt.savefig("week4_actual_vs_predicted.png")

plt.show()

print("\nActual vs Predicted graph saved successfully!")
# Step 11: Feature Importance Analysis

print("\n========== FEATURE IMPORTANCE ==========")

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": tree_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# Save feature importance
feature_importance.to_csv(
    "week4_feature_importance.csv",
    index=False
)

# Feature importance visualization
plt.figure(figsize=(9, 5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance - Decision Tree Regression")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("week4_feature_importance.png")

plt.show()

print("\nFeature importance analysis completed!")
# Step 12: Predictive Insights

print("\n========== PREDICTIVE INSIGHTS ==========")

print("\n1. Linear Regression achieved an R-squared score of "
      f"{r2_linear:.4f} with MAE of {mae_linear:.4f} hours.")

print("2. Decision Tree Regression achieved an R-squared score of "
      f"{r2_tree:.4f} with MAE of {mae_tree:.4f} hours.")

print("3. Distance_km has the highest feature importance in the "
      "Decision Tree model.")

print("4. Delay_Time_hr is the second most important feature in "
      "the Decision Tree model.")

print("5. Expected_Delivery_Time_hr is the third important feature "
      "in the Decision Tree model.")

print("6. The predictive results can be used to support delivery "
      "planning, delay monitoring, and route management.")
# Step 13: Optimization Recommendations

print("\n========== OPTIMIZATION RECOMMENDATIONS ==========")

print("\n1. Route Optimization:")
print("Prioritize analysis of long-distance routes because Distance_km "
      "has the highest feature importance.")

print("\n2. Delay Management:")
print("Monitor Delay_Time_hr closely and investigate recurring causes "
      "of delivery delays.")

print("\n3. Delivery Planning:")
print("Use Expected_Delivery_Time_hr together with route information "
      "to improve delivery scheduling.")

print("\n4. Vehicle Planning:")
print("Use vehicle capacity and utilization information when assigning "
      "vehicles to delivery orders.")

print("\n5. Cost Monitoring:")
print("Monitor fuel and transportation costs alongside delivery "
      "performance to identify cost optimization opportunities.")

print("\nOptimization recommendations completed successfully!")
# Step 14: Save Predictive Analysis Results

print("\n========== SAVING WEEK 4 RESULTS ==========")

# Save model performance
model_results.to_csv(
    "week4_model_performance.csv",
    index=False
)

# Save feature importance
feature_importance.to_csv(
    "week4_feature_importance.csv",
    index=False
)

# Create prediction results
prediction_results = pd.DataFrame({
    "Actual_Delivery_Time_hr": y_test.values,
    "Linear_Regression_Prediction": y_pred_linear,
    "Decision_Tree_Prediction": y_pred_tree
})

prediction_results.to_csv(
    "week4_prediction_results.csv",
    index=False
)

print("\nSaved files:")
print("1. week4_model_performance.csv")
print("2. week4_feature_importance.csv")
print("3. week4_prediction_results.csv")

print("\nWeek 4 predictive analysis results saved successfully!")
# Step 15: Final Summary

print("\n========== WEEK 4 FINAL SUMMARY ==========")

print("\nDataset Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nLinear Regression:")
print(f"MAE: {mae_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")
print(f"R²: {r2_linear:.4f}")

print("\nDecision Tree Regression:")
print(f"MAE: {mae_tree:.4f}")
print(f"RMSE: {rmse_tree:.4f}")
print(f"R²: {r2_tree:.4f}")

print("\nMost Important Decision Tree Feature:")
print(feature_importance.iloc[0]["Feature"])

print("\nWeek 4 analysis completed successfully!")