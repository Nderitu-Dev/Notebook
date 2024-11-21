# Importing required libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the Iris dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(url, header=None, names=column_names)

# Data Exploration
print("First 5 rows:")
print(df.head())

print("\nCheck for missing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

print("\nInfo about the dataset:")
df.info()

# Basic Data Analysis: Grouping by Species
print("\nGroup by species:")
print(df.groupby('species').mean())

# Data Visualization: Histogram of Sepal Length
plt.figure(figsize=(8,6))
plt.hist(df['sepal_length'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Pairplot: Relationships between features
sns.pairplot(df, hue='species')
plt.show()

# Boxplot: Sepal Length by Species
plt.figure(figsize=(8,6))
sns.boxplot(x='species', y='sepal_length', data=df)
plt.title('Boxplot of Sepal Length by Species')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
corr = df.drop('species', axis=1).corr()  # Exclude non-numeric column 'species'
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
