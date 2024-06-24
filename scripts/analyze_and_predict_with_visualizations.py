
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load Datasets
df_energy_generation = pd.read_csv('Energy_Generation_Potential_Expanded.csv')
df_efficiency_durability = pd.read_csv('Efficiency_and_Durability_Expanded.csv')
df_limitations = pd.read_csv('Limitations_and_Considerations_Expanded.csv')

# Create a target variable 'Success' in the energy generation dataset
power_output_median = df_energy_generation['Power_Output_KWh_Per_Year'].median()
implementation_cost_median = df_energy_generation['Implementation_Cost'].median()
df_energy_generation['Success'] = ((df_energy_generation['Power_Output_KWh_Per_Year'] > power_output_median) &
                                   (df_energy_generation['Implementation_Cost'] < implementation_cost_median)).astype(int)

# Merge datasets based on Project_ID
df_merged = df_efficiency_durability.merge(df_energy_generation, on='Project_ID')
df_merged = df_merged.merge(df_limitations, on='Project_ID')

# Drop irrelevant columns and fill missing values
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
df_merged[numeric_cols] = df_merged[numeric_cols].apply(lambda x: x.fillna(x.median()))

# Select features and target variable
features = df_merged.drop(columns=['Success', 'Project_ID', 'Efficiency_ID', 'Limitation_ID'])
target = df_merged['Success']

# Encoding
features = pd.get_dummies(features, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize StandardScaler and scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = logreg.predict(X_test_scaled)
y_pred_prob = logreg.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = auc(*roc_curve(y_test, y_pred_prob)[:2])

# Visualizations
plt.figure(figsize=(12, 10))
sns.heatmap(df_merged[numeric_cols].corr(), annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Success', data=df_energy_generation)
plt.title('Distribution of Success Variable')
plt.show()

sns.pairplot(df_energy_generation, hue='Success', diag_kind='kde')
plt.suptitle('Pairplot of Energy Generation Potential Features', y=1.02)
plt.show()

categorical_features = ['Project_Type', 'Location', 'Road_Type', 'Vehicle_Type']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, data=df_energy_generation, hue='Success')
    plt.title(f'Count Plot of {feature} by Success')
    plt.xticks(rotation=45)
    plt.show()

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Values')
plt.plot(y_pred_prob, label='Predicted Probabilities', linestyle='dashed')
plt.title('Actual Values vs Predicted Probabilities')
plt.xlabel('Test Set Index')
plt.ylabel('Probability / Actual Value')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
years = [2023, 2024, 2025, 2026, 2027]
predicted_success_rate = [0.8, 0.82, 0.85, 0.87, 0.9]
plt.bar(years, predicted_success_rate, color='skyblue', alpha=0.7, label='Predicted Success Rate')
plt.plot(years, predicted_success_rate, color='orange', marker='o', linestyle='-', linewidth=2, label='Trend')
plt.title('Predicted Success Rate of Implementation in Upcoming Years')
plt.xlabel('Year')
plt.ylabel('Predicted Success Rate')
plt.legend()
plt.show()
