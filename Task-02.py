#the following code is exploratory data analysis (EDA) on a wine quality dataset
#dataset from kaggle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine quality dataset
df = pd.read_csv("winequality-red.csv")

# Print the first few rows of the dataset
print(df.head())

# Get the shape of the dataset
print(df.shape)

# Get information about the dataset
df.info()

# Calculate summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values by filling with the mean
df.fillna(df.mean(), inplace=True)

# Calculate the correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.show()

# Create scatter plots of variables
sns.pairplot(df, x_vars=['fixed acidity', 'volatile acidity', 'citric acid'], y_vars='quality')
plt.show()

# Create box plots of variables
df.plot(kind='box', subplots=True, layout=(1, 3), figsize=(12, 6), 
        sharex=False, sharey=False, 
        column=['fixed acidity', 'volatile acidity', 'citric acid'])
plt.show()

# Create density plots of variables
plt.figure(figsize=(12, 6))
sns.kdeplot(df['fixed acidity'], shade=True)
sns.kdeplot(df['volatile acidity'], shade=True)
sns.kdeplot(df['citric acid'], shade=True)
plt.show()

