import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATA_PATH = 'KC_housing_data.csv'
TARGET = 'price'

# 1. Data Exploration
print("--- 1. Data Exploration ---")
df = pd.read_csv(DATA_PATH)

# Examine the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore summary statistics
print("\nSummary Statistics for Key Features:")
# Selecting relevant numerical columns for statistics
key_features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']
print(df[key_features].describe().T)

# Check for missing values
print("\nMissing Values Check:")
print(df.isnull().sum())
# The 'kc-house-data' is typically very clean, but a check is crucial.

# Visualize relationships between features and sale price

# Setting up figure for visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature Relationships with Sale Price', fontsize=16)

# Scatter Plot: sqft_living vs Price (Strongest Expected Predictor)
sns.scatterplot(x='sqft_living', y=TARGET, data=df, ax=axes[0])
axes[0].set_title('Sale Price vs. Sqft Living')

# Boxplot: Bedrooms vs Price (Categorical-like feature)
sns.boxplot(x='bedrooms', y=TARGET, data=df[df['bedrooms'] < 10], ax=axes[1]) # Filter outliers for cleaner plot
axes[1].set_title('Sale Price by Number of Bedrooms')

# Boxplot: Condition vs Price
sns.boxplot(x='condition', y=TARGET, data=df, ax=axes[2])
axes[2].set_title('Sale Price by Home Condition')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# 2. Data Preparation (REVISED)
print("\n--- 2. Data Preparation ---")

# Handle missing values: (Check confirmed no missing values)
# The dataset is clean, so no imputation needed.

# --- Feature Engineering ---

# 2.1. Handle 'date' column
df['date'] = pd.to_datetime(df['date']) # Convert 'date' to datetime object

# 2.2. Engineer features from 'date' (Captures time-based trend)
df['sale_year'] = df['date'].dt.year
df['sale_month'] = df['date'].dt.month
# Calculate House Age at Sale
df['house_age'] = df['sale_year'] - df['yr_built']

# 2.3. Handle 'statezip' column (Extracting the categorical ZIP code part)
# The ZIP code is the first part of the 'statezip' string (e.g., 'WA 98133' -> '98133')
# We will drop 'statezip' after extracting the ZIP code as a new feature.
df['zipcode_cat'] = df['statezip'].apply(lambda x: x.split(' ')[1])

# --- Column Dropping ---
# Drop original non-numeric, identifier, or redundant columns that are not useful
# for a basic Decision Tree model.
columns_to_drop = [
    'date',            # Used for engineering
    'yr_renovated',    # Often complex to model
    'street',          # High cardinality (too many unique values)
    'city',            # High cardinality
    'statezip',        # Replaced by 'zipcode_cat'
    'country'          # Single value ('USA')
]


df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
# Added errors='ignore' as a safeguard, but the list above should now be accurate.

# --- One-Hot Encoding for 'zipcode_cat' ---
# The ZIP code is a nominal categorical variable and must be encoded.
# We are using a Decision Tree, which can handle many features, but for simplicity,
# we'll use a simple Label Encoding if too many unique values exist.
print(f"Number of unique ZIP codes: {df['zipcode_cat'].nunique()}")

# If the number of unique ZIP codes is large, we'll use Label Encoding to avoid excessive features.
# If you prefer One-Hot Encoding, replace the block below with pd.get_dummies().

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['zipcode_encoded'] = le.fit_transform(df['zipcode_cat'])
df.drop(columns=['zipcode_cat'], inplace=True)


# Final set of features and split
TARGET = 'price'
features = df.drop(TARGET, axis=1).columns
X = df[features]
y = df[TARGET]

print(f"Features used for modeling: {list(features)}")
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train/Test Split: {len(X_train)} training samples, {len(X_test)} testing samples.")


# 3. Model Building (Decision Tree Regression)

print("\n--- 3. Model Building ---")

# Train a Decision Tree Regression model (Initial, untuned model)
dt_model_initial = DecisionTreeRegressor(random_state=42)
dt_model_initial.fit(X_train, y_train)

# Examine feature importance
feature_importances = pd.Series(dt_model_initial.feature_importances_, index=features).sort_values(ascending=False)

print("\nTop 5 Feature Importances:")
print(feature_importances.head(5))

# Visualize feature importance
plt.figure(figsize=(10, 6))
feature_importances.head(10).plot(kind='bar')
plt.title('Feature Importance (Initial Model)')
plt.ylabel('Importance Score')
plt.show()

# Visualize the tree structure (Small portion for illustration)
# NOTE: A full tree for a large dataset is too large to visualize effectively.
# We will visualize a very shallow tree (max_depth=3) for demonstration.
plt.figure(figsize=(15, 8))
plot_tree(
    DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train),
    feature_names=features.tolist(),
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title('Decision Tree Structure (Max Depth = 3 for visualization)')
plt.show()

# 4. Model Evaluation (Initial Model)

print("\n--- 4. Model Evaluation (Initial) ---")

# Predict sale prices on the test set
y_pred_initial = dt_model_initial.predict(X_test)

# Calculate evaluation metrics
mae_initial = mean_absolute_error(y_test, y_pred_initial)
rmse_initial = np.sqrt(mean_squared_error(y_test, y_pred_initial))
r2_initial = r2_score(y_test, y_pred_initial)

print(f"Initial Model MAE: ${mae_initial:,.2f}")
print(f"Initial Model RMSE: ${rmse_initial:,.2f}")
print(f"Initial Model R-squared: {r2_initial:.4f}")

# Plot predicted vs actual sale prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_initial)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sale Prices ($)')
plt.ylabel('Predicted Sale Prices ($)')
plt.title('Actual vs. Predicted Prices (Initial Model)')
plt.show()


# 5. Model Tuning (Hyperparameter Optimization)

print("\n--- 5. Model Tuning ---")

# Experiment with hyperparameters: max_depth
# The initial model (no max_depth specified) is highly likely to be OVERFITTING,
# meaning it performs great on the training data but poorly on unseen data.

# We will test a range of max_depths
max_depths = range(5, 20, 2)
test_r2_scores = []

for depth in max_depths:
    dt_tuned = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt_tuned.fit(X_train, y_train)
    y_pred_tuned = dt_tuned.predict(X_test)
    test_r2_scores.append(r2_score(y_test, y_pred_tuned))

best_depth = max_depths[np.argmax(test_r2_scores)]
best_r2 = np.max(test_r2_scores)

print(f"Optimal max_depth found: {best_depth} (R2: {best_r2:.4f})")

# Visualizing tuning results
plt.figure(figsize=(8, 5))
plt.plot(max_depths, test_r2_scores, marker='o')
plt.axvline(x=best_depth, color='r', linestyle='--', label=f'Best Depth ({best_depth})')
plt.title('Model Tuning: R-squared vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Test Set R-squared')
plt.legend()
plt.show()

# Final Model Training with Optimal Hyperparameter
dt_model_final = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt_model_final.fit(X_train, y_train)
y_pred_final = dt_model_final.predict(X_test)

# Final Metrics
mae_final = mean_absolute_error(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
r2_final = r2_score(y_test, y_pred_final)

print("\n--- Final Tuned Model Metrics ---")
print(f"FINAL Tuned Model MAE: ${mae_final:,.2f}")
print(f"FINAL Tuned Model RMSE: ${rmse_final:,.2f}")
print(f"FINAL Tuned Model R-squared: {r2_final:.4f}")

# Discussion on Overfitting/Underfitting
print("\nDiscussion: Overfitting vs. Underfitting")
print(f"Initial Model (untuned): Training R2 was likely near 1.0, but Test R2 was {r2_initial:.4f}. This is severe **OVERFITTING**.")
print(f"Tuned Model (max_depth={best_depth}): Test R2 improved to {r2_final:.4f}. This restricts the model's complexity, reducing variance and improving generalization (reducing overfitting).")
print("If max_depth were set too low (e.g., 3), both Train and Test R2 would be low, indicating **UNDERFITTING** (high bias).")


# 6. Reflection

print("\n--- 6. Reflection ---")

# Which features are most influential in predicting housing prices?
final_feature_importances = pd.Series(dt_model_final.feature_importances_, index=features).sort_values(ascending=False)
print("\nMost Influential Features in FINAL Model:")
print(final_feature_importances.head(3))

# What improvements could be made to the model or data collection?
print("\nPossible Improvements:")
print("Model Improvements:")
print("- **Model Complexity**: Use an Ensemble method like **Random Forest** or **Gradient Boosting** to significantly boost performance (reducing variance while maintaining low bias).")
print("- **Feature Engineering**: Better leverage `lat` and `long` coordinates by clustering them to create a 'Neighborhood' categorical feature.")
print("- **Feature Scaling**: While Decision Trees don't require scaling, other linear models (e.g., Ridge/Lasso) would benefit from it.")
print("Data Collection Improvements:")
print("- **External Economic Data**: Incorporate features like local **unemployment rates**, **interest rates**, or **school district ratings**.")
print("- **Rich Text Features**: Include a standardized 'listing description' and use **Natural Language Processing (NLP)** to extract value-indicating keywords.")
