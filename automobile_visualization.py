import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Load Automobile Dataset
# ===========================

df = pd.read_csv("Automobile.csv")

# ===========================
# Display Dataset
# ===========================

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# ===========================
# Visualization 1
# Price Distribution
# ===========================

plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=20)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# ===========================
# Visualization 2
# Fuel Type
# ===========================

plt.figure(figsize=(6,5))
sns.countplot(x="fuel-type", data=df)
plt.title("Fuel Type Distribution")
plt.show()

# ===========================
# Visualization 3
# Body Style
# ===========================

plt.figure(figsize=(8,5))
sns.countplot(x="body-style", data=df)
plt.title("Body Style Distribution")
plt.xticks(rotation=20)
plt.show()

# ===========================
# Visualization 4
# Horsepower Distribution
# ===========================

plt.figure(figsize=(8,5))
plt.hist(df["horsepower"], bins=20)
plt.title("Horsepower Distribution")
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.show()

# ===========================
# Visualization 5
# Box Plot
# ===========================

plt.figure(figsize=(8,5))
sns.boxplot(x=df["price"])
plt.title("Price Box Plot")
plt.show()

# ===========================
# Visualization 6
# Scatter Plot
# ===========================

plt.figure(figsize=(8,5))
plt.scatter(df["horsepower"], df["price"])
plt.title("Horsepower vs Price")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.show()

# ===========================
# Visualization 7
# Correlation Heatmap
# ===========================

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()