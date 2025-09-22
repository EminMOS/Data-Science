# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv("houses.csv")  
print(df.head())
print(df.columns)

# Step 3: Filter properties with basements
df_basement = df[df['sqft_basement'] > 0]
print(f"Total properties with basements: {len(df_basement)}")

# Step 4: Market potential analysis per city
city_summary = df_basement.groupby('city').agg(
    avg_price=('price', 'mean'),
    num_properties=('price', 'size'),  
    avg_sqft_living=('sqft_living', 'mean'),
    avg_sqft_basement=('sqft_basement', 'mean')
).reset_index()

# Step 5: Barplot - Average price per city
plt.figure(figsize=(12, 6))
sns.barplot(x='city', y='avg_price', data=city_summary, palette='Blues_d')
plt.title('Average Price per City (Properties with Basements)')
plt.xlabel('City')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()

# Step 6: Barplot - Number of properties per city
plt.figure(figsize=(12, 6))
sns.barplot(x='city', y='num_properties', data=city_summary, palette='Greens_d')
plt.title('Number of Properties per City (with Basements)')
plt.xlabel('City')
plt.ylabel('Number of Properties')
plt.xticks(rotation=45)
plt.show()

# Step 7: Scatter plot - Living area vs. Price, colored by city
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_basement, x='sqft_living', y='price', hue='city', palette='tab10')
plt.title('Living Area vs Price (Properties with Basements)')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Price')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Step 8: Heatmap - correlation between numeric features
numeric_cols = ['price', 'sqft_living', 'sqft_basement', 'bedrooms', 'bathrooms', 'floors', 'sqft_lot']
plt.figure(figsize=(10, 6))
sns.heatmap(df_basement[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Properties with Basements)')
plt.show()

# Step 9: Highlight cities with multiple affordable properties (4-5+) for "client clusters"
affordable_threshold = 500000 
city_affordable = df_basement[df_basement['price'] <= affordable_threshold].groupby('city').agg(
    num_affordable=('price', 'size')
).reset_index()

# Filter cities with at least 4 properties
potential_cities = city_affordable[city_affordable['num_affordable'] >= 4]
print("Cities with 4 or more affordable basement properties:")
print(potential_cities)