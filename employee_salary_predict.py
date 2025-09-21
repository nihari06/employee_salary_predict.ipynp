import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("adult 3.csv")

# Clean and encode
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['income'] = df['income'].apply(lambda x: 0 if x.strip() == '<=50K' else 1)
df.rename(columns={'income': 'salary'}, inplace=True)

# -----------📊 R² Score Evaluation Section -----------

# Prepare features and target
X = pd.get_dummies(df.drop('salary', axis=1), drop_first=True)
y = df['salary']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42)
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"{name}: R² Score = {r2:.4f}")
    results.append({'Model': name, 'R2 Score': round(r2, 4)})

# R² Score Bar Plot
results_df = pd.DataFrame(results)
plt.figure(figsize=(8, 5))
bars = plt.bar(results_df["Model"], results_df["R2 Score"], color='skyblue')
plt.title("Model R² Score Comparison")
plt.ylabel("R² Score")
plt.xlabel("Model")
plt.ylim(0, 1)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha='center')

plt.tight_layout()
plt.show()

# -----------📦 Boxplot After Outlier Removal -----------

# Select numeric columns for boxplot
numeric_cols = ['age', 'educational-num', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'salary']

# Function to remove outliers using IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

# Remove outliers for numeric columns
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

# Plot boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot After Outlier Removal")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
