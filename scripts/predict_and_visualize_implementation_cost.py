
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

# Initialize and train the random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Make Predictions
rf_y_pred = rf_clf.predict(X_test_scaled)
rf_y_pred_prob = rf_clf.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)

# Visualizations
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_energy_generation, x='Project_Start_Year', y='Implementation_Cost', hue='Success', palette='Set3')
plt.title('Implementation Cost vs Project Success Along with Years')
plt.xlabel('Project Start Year')
plt.ylabel('Implementation Cost')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(rf_y_pred_prob, bins=10, alpha=0.5, label='Random Forest', color='green')
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend()
plt.show()
