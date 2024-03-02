import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#1. Exploratory Data Analysis (EDA):
# TO Load the Titanic dataset
df = pd.read_csv('Dataset/titanic/train.csv')

# Number of rows, columns, and data types
rows, columns = df.shape
data_types = df.dtypes
print("\n\n")
print("TASKS:")
print("Describe the dataset: number of rows, columns, data types.")
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")
print("Data types:")
print(data_types)
print("--------------------------------")

# Summary statistics

print("\n")
print("Summarize numerical features:")
summary_stats = df.describe()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Calculate summary statistics
summary_statistics = pd.DataFrame({
    'Mean': df[numeric_columns].mean(),
    'Median': df[numeric_columns].median(),
    'Mode': df[numeric_columns].mode().iloc[0],  # Mode can have multiple values, we choose the first
    'Range': df[numeric_columns].max() - df[numeric_columns].min(),
    'Standard Deviation': df[numeric_columns].std()
})

# print(summary_statistics)
print("--------------------------------")



print("\n")
print("Tasks:Explore categorical features: frequency distribution, unique values.")
# Select only categorical columns
categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    print(f"\nExploring Categorical Feature: {column}")
    print(f"Frequency Distribution:\n{df[column].value_counts()}\n")
    print(f"Unique Values: {df[column].unique()}\n")

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns



# Visualize distributions using histograms
plt.figure(figsize=(15, 10))
df[numeric_columns].hist(bins=20, grid=False)
plt.suptitle("Histograms of Numerical Features", y=1.02)
plt.show()


print("\n")
print("Tasks:Visualize distributions: histograms, box plots.")
#Visualize distributions using box plots
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns])
plt.title("Box Plots of Numerical Features")
plt.show()


print("\n")
print("Identify outliers and handle them appropriately.")
columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
# Identify outliers using z-score
z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())
outliers = (z_scores > 3).any(axis=1)

#display the number of outliers

print("Number of outliers:", outliers.sum())

# Handling outliers (replace with median)
df_no_outliers = df.copy()
df_no_outliers[numeric_columns] = df_no_outliers[numeric_columns].mask(outliers, df_no_outliers[numeric_columns].median(),axis=0)


print("\nNumerical Features Summary without Outliers:")
print(df_no_outliers[numeric_columns].describe())

#-----------------------------------------------------------------------------------------------------------------------------------------------
#2. Handling Missing Values:
# Count missing values per column

print("\n")
print("Tasks:Identify missing values: count missing values per column.")
missing_values = df.isnull().sum()

print("Missing values per column:")
print(missing_values)


print("\n")
print("Tasks:Evaluate the impact of missing data:.")
# missing_age_corr = df['Age'].isnull().corr(df['Pclass'])
# print(f"Correlation between missing Age values and Pclass: {missing_age_corr}")

# Imputation using mean, median, or mode
df['Age'].fillna(df['Age'].median(), inplace=True)

#-----------------------------------------------------------------------------------------------------------------------------------------------

# 3.Dealing with duplicate values
# Detect duplicate rows based on all columns
duplicates = df.duplicated()

# Analyze frequency and distribution of duplicates
duplicate_counts = duplicates.value_counts()
print("Frequency of duplicate values:")
print(duplicate_counts)

#remove duplicates
df.drop_duplicates(inplace=True)


# Assess impact on data integrity
new_rows, new_columns = df.shape
print(f"Number of rows after removing duplicates: {new_rows}")
print(f"Number of columns: {new_columns}")
