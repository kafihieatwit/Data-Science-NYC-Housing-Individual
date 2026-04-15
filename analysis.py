# Individual Project - Emraan Kafihi
# Data Science Fundamentals - COMP 3125

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("Dataset/nyc_housing_base.csv")

# Drop missing values
df = df.dropna()

# Convert data types
df["zip_code"] = df["zip_code"].astype(int)
df["yearbuilt"] = df["yearbuilt"].astype(int)
df["unitsres"] = df["unitsres"].astype(int)
df["unitstotal"] = df["unitstotal"].astype(int)
df["numfloors"] = df["numfloors"].astype(int)
df["landuse"] = df["landuse"].astype(int)
df["building_age"] = df["building_age"].astype(int)

# Feature engineering
df = df[df['bldgarea'] > 0]
df['price_per_sqft'] = df['sale_price'] / df['bldgarea']
df = df[df['price_per_sqft'] < df['price_per_sqft'].quantile(0.99)]

# Individual RQ1: Does price per square foot vary by building size?
bins = [0, 1000, 2500, 10000, 50000, df['bldgarea'].max()]
labels = ['Small\n(<1k)', 'Medium\n(1k-2.5k)', 'Large\n(2.5k-10k)', 'Very Large\n(10k-50k)', 'Massive\n(50k+)']
df['size_group'] = pd.cut(df['bldgarea'], bins=bins, labels=labels)

size_stats = df.groupby('size_group', observed=True)['price_per_sqft'].agg(
    median=lambda x: np.median(x),
    mean=lambda x: np.mean(x)
).reset_index()

plt.bar(size_stats['size_group'].astype(str), size_stats['median'], color='steelblue', edgecolor='white')
plt.xlabel('Building Size (sq ft)')
plt.ylabel('Median Price per Sq Ft ($)')
plt.title('Median Price per Square Foot by Building Size')
plt.tight_layout()
plt.savefig('figures/price_per_sqft_by_size.png')
plt.show()

# Individual RQ2: What properties are the strongest predictors of sale price?
cols = ['bldgarea', 'lotarea', 'resarea', 'numfloors', 'unitsres', 'unitstotal', 'building_age']
corr = df[cols + ['sale_price']].corr()['sale_price'].drop('sale_price')
corr_sorted = corr.abs().sort_values(ascending=False)
corr_values = corr[corr_sorted.index]

colors = ['steelblue' if v >= 0 else 'tomato' for v in corr_values]
plt.bar(corr_values.index, corr_values.values, color=colors, edgecolor='white')
plt.xlabel('Property Feature')
plt.ylabel('Correlation with Sale Price')
plt.title('Correlation of Property Features with Sale Price')
plt.xticks(rotation=30, ha='right')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('figures/feature_correlation_sale_price.png')
plt.show()