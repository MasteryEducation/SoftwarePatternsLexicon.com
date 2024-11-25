---
linkTitle: "Data Wrangling"
title: "Data Wrangling: Cleaning and Transforming Data"
description: "Cleaning and transforming data from raw form to a structured format"
categories:
- Data Management Patterns
tags:
- Data Wrangling
- Data Cleaning
- Data Transformation
- ETL
- Preprocessing
date: 2024-07-07
type: docs
canonical: "https://softwarepatternslexicon.com/machine-learning/data-management-patterns/data-integration/data-wrangling"
license: "© 2024 Tokenizer Inc. CC BY-NC-SA 4.0"
---

## Introduction

Data Wrangling, also referred to as data munging, is the process of cleaning and transforming raw data into a structured and useful format suitable for analysis. This is a critical step in the data science and machine learning pipeline because the quality of the data directly influences the quality of the insights derived from it.

Data Wrangling is part of Data Integration, which falls under Data Management Patterns. This pattern is essential for handling inconsistencies, missing values, and formatting issues inherent in raw data.

## Key Objectives
The primary objectives of Data Wrangling include:

1. **Cleaning Data**: Removing duplicates, correcting errors, and handling missing values.
2. **Transforming Data**: Normalizing, aggregating, and enriching the data for better analysis.
3. **Structuring Data**: Converting raw data into a structured form like CSV, JSON, or database formats.

## Process Overview
The Data Wrangling process typically involves:

1. **Data Collection**: Gathering raw data from various sources.
2. **Data Cleaning**: Identifying and correcting errors and inconsistencies.
3. **Data Transformation**: Converting data into a useful format.
4. **Data Integration**: Merging data from various sources into a coherent dataset.
5. **Data Validation**: Ensuring the data meets the required quality standards.

## Coding Example

Here, we outline simple examples for Data Wrangling in Python using Pandas, and in R using dplyr.

### Example in Python (Pandas)

```python
import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', np.nan],
    'Age': [25, 30, 35, np.nan, 40],
    'Salary': [50000, np.nan, 70000, 80000, 60000]
}

df = pd.DataFrame(data)

print("Raw Data:\n", df)

df.dropna(subset=['Name'], inplace=True)  # Dropping rows where Name is NaN
df['Age'].fillna(df['Age'].mean(), inplace=True)  # Filling NaN values in Age with mean age
df['Salary'].fillna(df['Salary'].median(), inplace=True)  # Filling NaN values in Salary with median salary

df['Salary'] = df['Salary'] * 1.1  # Example transformation: increasing Salary by 10%

df.rename(columns={'Name': 'Employee Name', 'Age': 'Age (Years)', 'Salary': 'Salary ($)'}, inplace=True)

print("\nCleaned Data:\n", df)
```

### Example in R (dplyr)

```r
library(dplyr)

data <- data.frame(
  Name = c('Alice', 'Bob', 'Charlie', 'David', NA),
  Age = c(25, 30, 35, NA, 40),
  Salary = c(50000, NA, 70000, 80000, 60000)
)

print("Raw Data:")
print(data)

data <- data %>%
  filter(!is.na(Name)) %>%  # Dropping rows where Name is NA
  mutate(
    Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age),  # Filling NA values in Age with mean age
    Salary = ifelse(is.na(Salary), median(Salary, na.rm = TRUE), Salary)  # Filling NA values in Salary with median salary
  )

data <- data %>%
  mutate(Salary = Salary * 1.1)  # Example transformation: increasing Salary by 10%

data <- data %>%
  rename('Employee Name' = Name, 'Age (Years)' = Age, 'Salary ($)' = Salary)

print("\nCleaned Data:")
print(data)
```

## Related Design Patterns

1. **ETL (Extract, Transform, Load)**:
   - A common technique in data warehousing to extract data from different sources, transform it into a consistent format, and load it into a target database.
   - **Description**: ETL is essential for integrating data from various sources and transforming it into a useful format for analysis.

2. **Data Validation**:
   - Ensuring the accuracy and quality of the data before using it for analysis.
   - **Description**: Involves checking for data integrity, consistency, and completeness.

3. **Pipelineization**:
   - Creating a sequence of data processing steps to automate repetitive data-related tasks.
   - **Description**: A pipeline integrates multiple data wrangling steps into an end-to-end workflow.

## Additional Resources
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [dplyr Documentation](https://dplyr.tidyverse.org/)
- [Scikit-Learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)

## Summary

Data Wrangling is a fundamental design pattern in machine learning and data science. It involves cleaning, transforming, and structuring raw data to make it ready for analysis. By meticulously handling missing values, correcting errors, and transforming data into a consistent format, Data Wrangling ensures that high-quality and reliable insights can be derived. Python's Pandas and R's dplyr packages are powerful tools to facilitate this critical preparatory step. Integrating Data Wrangling with other design patterns like ETL, Data Validation, and Pipelineization strengthens the overall data management strategy and enhances analytic capabilities.
