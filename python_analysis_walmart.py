import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.preprocessing import LabelEncoder

# now load the data
df_features = pd.read_csv('features.csv')
df_departments = pd.read_csv('departments.csv', sep=';')
df_stores = pd.read_csv('stores.csv')
df_train = pd.read_csv('train.csv')

# merge dataframes into df
df_merged = df_train.merge(df_features, on= ['Store', 'Date'], how = 'left')
df_merged = df_stores.merge(df_merged, on = ['Store'], how = 'right')
df_merged = df_departments.merge(df_merged, on = ['Dept'], how = 'right')

print(df_merged.head())

# convert date to datetime format
df_merged['Date'] = pd.to_datetime(df_merged['Date'])

#drop one isholiday column
df_merged = df_merged.drop(['IsHoliday_y'], axis = 1)

#rename empty description values
df_merged['Description'] = df_merged['Description'].fillna('No Description')

# we need group means for temp, fiuel price, cpi, unemployment, weekly sales. 
#categoeries are store, dept, date, isholiday, type, size, description

#for the outliers, we'll consider weekly sales, temperature, fuel price, cpi
outliers_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

#lets count the outliers and store them in a dictionary
outliers = {}

df_clean = df_merged.copy()

for column in outliers_columns:
    Q1 = df_merged[column].quantile(0.25)
    Q3 = df_merged[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers= df_merged[(df_merged[column] < (Q1 - 1.5 * IQR)) | (df_merged[column] > (Q3 + 1.5 * IQR))]

df_clean = df_clean[~df_clean.index.isin(outliers.index)]

group_columns = ['Store', 'Dept', 'Type', 'Description']
average_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']

#calc means for the three columns grouped by dept
grouped_mean = df_clean.groupby(group_columns)[average_columns].mean()
print(grouped_mean.head())

#howmany outliers are there?
print(outliers.count())
print(outliers.info())

#what percentage of the data is outliers?
print(outliers.count()/df_merged.count())


#checking distributions for our numerical columns
sns.histplot(df_clean['Weekly_Sales'])
plt.show()

sns.histplot(df_clean['Temperature'])
plt.show()

sns.histplot(df_clean['Fuel_Price'])
plt.show()

sns.histplot(df_clean['CPI'])
plt.show()

sns.histplot(df_clean['Unemployment'])
plt.show()

sns.histplot(df_clean['Size'])
plt.show()

#interpretation:
#Weekly sales is not normally distributed
#Temperature is normally distributed
#Fuel price is not normally distributed
#CPI is not normally distributed
#Unemployment is not normally distributed
#Size is not normally distributed


#checking range, mean, median, and mode for each numerical column (without describe method)
#Weekly sales
print('Range:', df_clean['Weekly_Sales'].max() - df_clean['Weekly_Sales'].min())
print('Mean:', df_clean['Weekly_Sales'].mean())
print('Median:', df_clean['Weekly_Sales'].median())
print('Mode:', df_clean['Weekly_Sales'].mode())

#Temperature
print('Range:', df_clean['Temperature'].max() - df_clean['Temperature'].min())
print('Mean:', df_clean['Temperature'].mean())
print('Median:', df_clean['Temperature'].median())
print('Mode:', df_clean['Temperature'].mode())

#Fuel Price
print('Range:', df_clean['Fuel_Price'].max() - df_clean['Fuel_Price'].min())
print('Mean:', df_clean['Fuel_Price'].mean())
print('Median:', df_clean['Fuel_Price'].median())
print('Mode:', df_clean['Fuel_Price'].mode())

#CPI
print('Range:', df_clean['CPI'].max() - df_clean['CPI'].min())
print('Mean:', df_clean['CPI'].mean())
print('Median:', df_clean['CPI'].median())
print('Mode:', df_clean['CPI'].mode())

#Unemployment
print('Range:', df_clean['Unemployment'].max() - df_clean['Unemployment'].min())
print('Mean:', df_clean['Unemployment'].mean())
print('Median:', df_clean['Unemployment'].median())
print('Mode:', df_clean['Unemployment'].mode())

#Size
print('Range:', df_clean['Size'].max() - df_clean['Size'].min())
print('Mean:', df_clean['Size'].mean())
print('Median:', df_clean['Size'].median())
print('Mode:', df_clean['Size'].mode())

#checking unique values for categorical columns
#Store
print(df_clean['Store'].unique())
#Dept
print(df_clean['Dept'].unique())
#Type
print(df_clean['Type'].unique())
#Description
print(df_clean['Description'].unique())

#checkign frequency of categorical columns
#Store
print(df_clean['Store'].value_counts())
#Dept
print(df_clean['Dept'].value_counts())
#Type
print(df_clean['Type'].value_counts())
#Description
print(df_clean['Description'].value_counts())

#for bivariate analysis, we'll compare the following columns:
#weekly sales and store
#weekly sales and dept
#weekly sales and type
#weekly sales and description
#weekly sales and isholiday

#weekly sales and store
sns.scatterplot(x='Weekly_Sales', y='Store', data=df_clean)
plt.show()

#weekly sales and dept
sns.scatterplot(x='Weekly_Sales', y='Dept', data=df_clean)
plt.show()

#weekly sales and type
sns.scatterplot(x='Weekly_Sales', y='Type', data=df_clean)
plt.show()

#weekly sales and description
sns.violinplot(x='Weekly_Sales', y='Description', data=df_clean)
plt.show()

#weekly sales and isholiday
sns.scatterplot(x='Weekly_Sales', y='IsHoliday_x', data=df_clean)
plt.show()

sns.countplot(x='Store', data=df_clean)
plt.show()



#Which categorical columns are you comparing against each other?
#Store and Dept
#Store and Type
#Store and Description
#Dept and Type
#Dept and Description
#Type and Description


#checkign unique combinations for categorical columns
#Store and Dept
print(df_clean.groupby(['Store', 'Dept']).size())

#Store and Type
print(df_clean.groupby(['Store', 'Type']).size())

#Store and Description
print(df_clean.groupby(['Store', 'Description']).size())

#Dept and Type
print(df_clean.groupby(['Dept', 'Type']).size())

#Dept and Description
print(df_clean.groupby(['Dept', 'Description']).size())

#Type and Description
print(df_clean.groupby(['Type', 'Description']).size())

print(df_clean.isnull().sum())

print(df_clean.describe())

#based on this it would be interesting to check sales on max CPI periods,
# or when unemployment is at max
# we can check other variables when Weekly Sales is negative
#visualizing the relationships with a mosaic plot

#tried making a mosaic plot for store and dept, but it was not very informative
mosaic(df_clean, ['Store', 'Dept'])
plt.show()

#lets try a countplot
sns.countplot(x='Store', hue='Type', data=df_clean)
plt.show()

#another countplot
sns.countplot(x='Dept', hue='Type', data=df_clean)
plt.show()

#now to check the correlation between the variables
correlation = df_clean.corr()
sns.heatmap(correlation, annot = True)
plt.show()

# Prerparing the data for Feature Engineering
#Breaking down the Date to Days/Weeks/Months
#We May not need 'day of week' because reporting is always done on fridays or 'year' depending
df_clean['Day_of_Week'] = pd.to_datetime(df_clean['Date']).dt.day_name()
df_clean['Month'] = pd.to_datetime(df_clean['Date']).dt.month
df_clean['Year'] = pd.to_datetime(df_clean['Date']).dt.year
        
print(df_clean['Year'].unique())
print(df_clean.info())

# we have to make sure to include Day, Department and Store in the Feature Matrix when we fit the model
#to bin the numerical columns:
#Temperature:
temp_bins = [0, 32, 70, 100]
temp_labels = ['Low', 'Medium', 'High']
df_clean['Temperature_binned'] = pd.cut(df_clean['Temperature'], bins = temp_bins, labels=temp_labels)

#Fuel_Price:
fuel_bins = [1, 2, 3, 4, 5]
fuel_labels = ['Low', 'Normal', 'High', 'Very High']
df_clean['Fuel_Price_binned'] = pd.cut(df_clean['Fuel_Price'], bins = fuel_bins, labels=fuel_labels)

#Unemployment:
print(df_clean['Unemployment'].describe())
unemp_bins = [4, 6, 8, 10]
unemp_labels = ['Low', 'Normal', 'High']
df_clean['Unemployment_binned'] = pd.cut(df_clean['Unemployment'], bins=unemp_bins, labels=unemp_labels)

#CPI:
print(df_clean['CPI'].describe())
cpi_bins= [110, 140, 170, 200, 230] 
cpi_labels= ['Low', 'Mean', 'High', 'Very High']
df_clean['CPI_binned'] = pd.cut(df_clean['CPI'], bins=cpi_bins, labels=cpi_labels)

#Size:
print(df_clean['Size'].describe())
size_bins= [30000, 60000, 90000, 120000, 150000, 220000]
size_labels= ['XS', 'S', 'M', 'L', 'XL']
df_clean['Size_binned']=pd.cut(df_clean['Size'], bins=size_bins, labels=size_labels)

#creating dummies for Department, Store, 
# i used labelencoder instead of dummies because of the large number of categories
#labelencoder is a better option for categorical variables with many categories
labelencoder = LabelEncoder()
df_clean['Dept'] = labelencoder.fit_transform(df_clean['Dept'])
df_clean['Store'] = df_clean['Store']=labelencoder.fit_transform(df_clean['Store'])

print(df_clean.head())

