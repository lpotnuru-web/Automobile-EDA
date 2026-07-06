import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ==========================
# PART (a): Exploratory Data Analysis (EDA)
# ==========================

# Display the first 5 rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Display the entire dataset
print("\nComplete Dataset:")
print(df)

# Display dataset information
print("\nDataset Information:")
df.info()

# Display the shape of the dataset
print("\nShape of Dataset (Rows, Columns):")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# EDA Visualization 1
# ==========================

sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# ==========================
# EDA Visualization 2
# ==========================

sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()

# ==========================
# PART (b): Identify Attribute Types
# ==========================

print("\n----- PART (b): Identify Attribute Types -----")

binary_attributes = ["Survived", "Sex"]
nominal_attributes = ["Name", "Ticket", "Cabin", "Embarked"]
numeric_attributes = ["PassengerId", "Age", "SibSp", "Parch", "Fare"]
ordinal_attributes = ["Pclass"]

print("Binary Attributes:", binary_attributes)
print("Nominal Attributes:", nominal_attributes)
print("Numeric Attributes:", numeric_attributes)
print("Ordinal Attribute:", ordinal_attributes)

# ==========================
# PART (c): Measure Dissimilarity for Nominal Attributes
# ==========================

print("\n----- PART (c): Nominal Attribute Dissimilarity -----")

# Compare Sex
sex1 = df.loc[0, "Sex"]
sex2 = df.loc[1, "Sex"]

print("\nComparing Sex:")
print("Passenger 1:", sex1)
print("Passenger 2:", sex2)

if sex1 == sex2:
    print("Sex Dissimilarity = 0 (Same)")
else:
    print("Sex Dissimilarity = 1 (Different)")

# Compare Embarked
emb1 = df.loc[0, "Embarked"]
emb2 = df.loc[1, "Embarked"]

print("\nComparing Embarked:")
print("Passenger 1:", emb1)
print("Passenger 2:", emb2)

if emb1 == emb2:
    print("Embarked Dissimilarity = 0 (Same)")
else:
    print("Embarked Dissimilarity = 1 (Different)")

# ==========================
# PART (d): Measure Dissimilarity for Numeric Attributes
# ==========================

print("\n----- PART (d): Numeric Attribute Dissimilarity -----")

age1 = df.loc[0, "Age"]
fare1 = df.loc[0, "Fare"]

age2 = df.loc[1, "Age"]
fare2 = df.loc[1, "Fare"]

print("\nPassenger 1")
print("Age :", age1)
print("Fare:", fare1)

print("\nPassenger 2")
print("Age :", age2)
print("Fare:", fare2)

# Euclidean Distance
distance = sqrt((age1 - age2)**2 + (fare1 - fare2)**2)

print("\nEuclidean Distance =", distance)