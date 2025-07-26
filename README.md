# **Report: Uber Fares Dataset Analysis**
- **Author:** Muheto Jess Rutayisire 
- **Student ID:** 26481
- **Course:** Introduction to Big Data Analytics (INSY 8413)
- **Instructor:** Prof. Eric Maniraguha
- **Date:** 2025-07-26

---
### **Introduction**

The Purpose of this project  was to conduct a comprehensive analysis of the Uber Fares Dataset. The goal was to gather insights about fares amounts and  ride patterns, and other important metrics. This  This involved many consecutive processes of data cleaning, feature engineering, and the use of an interactive Power BI dashboard to visualize the findings.

Initially, the raw dataset was processed using Python for extensive cleaning and preparation; then Useful features were engineering (in Python) to enable further analysis. The enhanced dataset was then imported into Microsoft Power BI, where an interactive dashboard was made to explore the data dynamically. The final output is a detailed and dynamic analytical report summarizing the findings made out of the data.

*   **Primary Tools:** Python(Jupyter Notebook) with (Pandas, NumPy, Matplotlib, Seaborn) and Microsoft Power BI.
*   **Dataset:** Uber Fares Dataset sourced from Kaggle.

### **Methodology**

The First step was loading, understanding, and cleaning the data using a Jupyter Notebook with Python.

First, essential libraries for data manipulation and visualization were imported. The dataset (`uber.csv`) was then loaded with a Pandas DataFrame.

```python
# Import the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset as a Dataframe and disply the first 10 rows
df = pd.read_csv("uber.csv")
df.head(10)
```

Basic Data analysis was performed on the DataFrame to get the structure, datatype and dimensions. These will used to identify issues that will be handled in the data cleaning  

```python
# Summary of the Dataframe
df.info()
# Describe basic statistics of the dataset
df.describe()
# Dimensions of the Dataset
df.shape
```

**Data Cleaning**
The data was  cleaned to handle missing values, unreasonable data points (e.g., zero fares, invalid coordinates, etc...), and incorrect data types.

```python
# Drop rows with missing values
df_cleaned = df.dropna()

# Remove unreasonable fare amounts
df_cleaned = df_cleaned[df_cleaned['fare_amount'] > 0]

# Remove unreasonable passenger count
df_cleaned = df_cleaned[df_cleaned['passenger_count'] < 20]

# Remove unreasonable coordinates  
df_cleaned = df_cleaned[(df_cleaned['pickup_longitude'] != 0) & 
        (df_cleaned['pickup_latitude'] != 0) & 
        (df_cleaned['dropoff_longitude'] != 0) & 
        (df_cleaned['dropoff_latitude'] != 0)]

# The date and time is converted to the right format and errors removed
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],errors='coerce')
df.dropna(subset=['pickup_datetime'], inplace=True)

# Reset index
df_cleaned.reset_index(drop=True, inplace=True)
```

The cleaned data was saved to a new CSV file (`cleaned_uber_fares.csv`) for record-keeping and further analysis.

```python
df_cleaned.to_csv('cleaned_uber_fares.csv', index=False)
```

##### **Exploratory Data Analysis (EDA)**

Descriptive statistics were generated, and box plots were used to visualize the distribution of `fare_amount` and `passenger_count` and to identify outliers.

```python
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('cleaned_uber_fares.csv')
print(df.describe())

# The date and time is converted to the right format and errors removed
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],errors='coerce')
```

***Box plots generated***
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# Box plot for fare_amount
plt.subplot(1, 2, 1)
sns.boxplot(y=df['fare_amount'])
plt.title('Box Plot of Fare Amount')
plt.ylim(0, 40)

# Box plot for passenger_count
plt.subplot(1, 2, 2)
sns.boxplot(y=df['passenger_count'])
plt.title('Box Plot of Passenger Count')
plt.show()
```
![Box_plot](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/output.png)

##### **Feature Engineering**
New features such as `distance`,`hour`, `day_of_week`, and `peak_offpeak` indicators were created to support a more detailed analysis in Power BI.

The Haversine formula was implemented to calculate the distance of each ride, 
```python
import numpy as np
# Function to calculate distance using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Earth radius in kilometers
    R = 6371  
    
    # Haversine calculation
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Calculate the distance for each ride
df['distance'] = haversine(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)
```

and time-based features were extracted.
```python
# The date and time is converted to the right format and errors removed
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],errors='coerce')
df.dropna(subset=['pickup_datetime'], inplace=True)

# Extract time
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.day_name()
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year
```

A histogram was created to show the distribution of fare amounts. 
```python
plt.figure(figsize=(10, 6))
sns.histplot(df['fare_amount'], bins=50, kde=True)
plt.title('Distribution of Uber Fare Amounts')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')
plt.xlim(0, 80) # Focusing on the most common fare range
plt.show()
```
![Histogram](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/output2.png)

The relationships between fare, distance, and time of day were then visualized.
```python
import matplotlib.pyplot as plt
import seaborn as sns

#Fare amount and distance
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance', y='fare_amount', data=df, alpha=0.5)
plt.title('Fare Amount vs. Distance Traveled')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount ($)')
plt.xlim(0, 50) # Remove extreme distance outliers for better visualization
plt.ylim(0, 150)
plt.show()

#Fare amount and time
plt.figure(figsize=(14, 7))
sns.boxplot(x='hour', y='fare_amount', data=df)
plt.title('Fare Amount vs. Time of Day (Hour)')
plt.xlabel('Hour of the Day (0-23)')
plt.ylabel('Fare Amount ($)')
plt.ylim(0, 30) # Limit y-axis to see the main distribution clearly
plt.show()

#Heat map with other correlations
corr = df[['fare_amount', 'distance', 'hour']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Heat map of correlations')
plt.show()
```
![scatterplot](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/output3.png)
![boxplot](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/output5.png)
![heatmap](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/output6.png)

The final, enriched dataset was exported as `enhanced_uber_fares.csv`, ready for import into Power BI.
```python
# Export the enhanced dataframe to a new CSV file
df.to_csv('enhanced_uber_fares.csv', index=False)
```

### Analysis & Results

**Fare Distribution:** The distribution of fares is right-skewed, with the vast majority of rides costing between $5 and $25.
![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/output2.png)

**Fare vs. Distance:** There is a strong, positive correlation between `distance_km` and `fare_amount`. This confirms the primary driver of cost is the distance traveled. Outliers exist where short-distance trips have high fares, potentially indicating surge pricing or wait times.
![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/output3.png)

#### **PowerBI Results**
Iniatial Data analysis with Power BI
![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20162805.png)

#### Final Dashboard
![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20163127.png)

*   **Hourly Trends:** Ride volume exhibits a clear bimodal pattern, peaking during the morning commute (around 8 AM) and again during the evening rush (5-7 PM). The highest peak occurs in the evening.
![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20163344.png)

*Peak hours button selected*

*   **Daily Trends:** Fridays and Saturdays are the busiest days of the week, indicating higher demand for leisure and social travel. Mid-week days (Tuesday-Thursday) show consistent commuter patterns.
![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20164305.png)

*Distance traveled and Amount spent during the week*

![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20164026.png)

*Data on Friday Rides (One of the busiest days)*

*   **Monthly Trends:** The data shows some seasonality, with ride volume generally increasing in the spring and autumn months.

![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20164551.png)

**Geospatial Analysis**
The map visualization shows that the highest concentration of Uber rides originates in dense urban centers, with Manhattan being a primary hotspot.

![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20165614.png)

*Map of the World showing concentration of Uber rides*

![](https://github.com/MuhetoJess/Assignment1_PowerBI/blob/main/Screenshots%20and%20Images/Screenshot%202025-07-26%20164951.png)

*Map of Manhattan, New York superimposed with ride data*

The dataset did not include drop_off timestamps. thus it is impossible to calculate the actual ride duration. Distance was used as a proxy for ride length.
Indeed Weather data was not available in this dataset. Therefore, its potential impact on fare patterns and ride volume could not be investigated.

### **Conclusion**
The Analysis shows predictable and actionable patterns within the Uber Fares dataset.
 Ride demand is not random; it follows distinct hourly, daily, and weekly cycles driven by specific social behavior. The price of fares is primarily by distance traveled. And Lastly the service is most heavily utilized in densely populated urban areas.

### **Recommendations**
There are some recommendations that could be made for Uber based on the analyzed data that could enhance operations and give Uber a competitive advantage

For Example, instead of just applying surge pricing to riders, Uber could offer targeted bonuses to drivers to be active in high-demand zones just before the predictable morning (7-9 AM) and evening (5-7 PM) peaks. This could help stabilize fare prices and reduce rider wait times.

Also Given that Fridays and Saturdays are the busiest days, Uber could introduce targeted marketing campaigns or "Weekend Rider" promotions to capture an even larger share of the leisure market and build customer loyalty.

More could be done with respect the Data collection. Indeed the lack of Dropoff Timestamps and Weather Data limits the possibility of data collection and therefore could stand to be made available in further analysis of Ubers operations.

