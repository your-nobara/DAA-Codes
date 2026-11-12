
# Assignment No. 1
# Predict Uber fare using Linear Regression and Random Forest

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# Importing the dataset
# --------------------------------------------------

df = pd.read_csv("uber.csv")

print(df)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

# --------------------------------------------------
# 1. Pre-process the dataset
# --------------------------------------------------

# Handling missing values
print(df.isnull().sum())

df["fare_amount"] = df["fare_amount"].fillna(
    df["fare_amount"].median()
)
df["pickup_longitude"] = df["pickup_longitude"].fillna(
    df["pickup_longitude"].median()
)
df["pickup_latitude"] = df["pickup_latitude"].fillna(
    df["pickup_latitude"].median()
)
df["dropoff_longitude"] = df["dropoff_longitude"].fillna(
    df["dropoff_longitude"].median()
)
df["dropoff_latitude"] = df["dropoff_latitude"].fillna(
    df["dropoff_latitude"].median()
)
df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)

print(df.isnull().sum())

# Convert pickup_datetime into datetime format
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
)

# Remove rows with invalid datetime values
df.dropna(subset=["pickup_datetime"], inplace=True)

# Extract date and time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["dayofweek"] = df["pickup_datetime"].dt.dayofweek

# Drop pickup_datetime column
df.drop("pickup_datetime", axis=1, inplace=True)

# Calculate distance between pickup and drop-off
# coordinates using the Haversine formula

def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return 6371 * c

df["dist_travel_km"] = haversine_distance(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"]
)

print(df.head())
print(df.dtypes)

# --------------------------------------------------
# 2. Identify the outliers
# --------------------------------------------------

# Plot boxplots for numerical columns
df.plot(
    kind="box",
    subplots=True,
    layout=(7, 2),
    figsize=(12, 20)
)
plt.tight_layout()
plt.show()

# Function to identify and handle outliers using IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR

    df[col] = df[col].clip(
        lower=lower_whisker,
        upper=upper_whisker
    )

    return df

def treat_outliers_all(df, col_list):
    for col in col_list:
        df = remove_outliers(df, col)

    return df

# Select numerical columns
numeric_cols = df.select_dtypes(
    include=["number"]
).columns

df = treat_outliers_all(df, numeric_cols)

# Plot boxplots after handling outliers
df.plot(
    kind="box",
    subplots=True,
    
    figsize=(12, 20)
)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. Check the correlation
# --------------------------------------------------

print(df.corr(numeric_only=True))

plt.figure(figsize=(12, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.show()

# --------------------------------------------------
# 4. Implement regression models
# --------------------------------------------------

# Dividing the dataset into features and target
X = df[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "hour",
        "day",
        "month",
        "year",
        "dayofweek",
        "dist_travel_km"
    ]
]

y = df["fare_amount"]

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print(X_train.head())

# Linear Regression
regression = LinearRegression()
regression.fit(X_train, y_train)

print("Linear Regression")
print("Intercept:", regression.intercept_)
print("Coefficients:", regression.coef_)

y_prediction = regression.predict(X_test)

print("Predicted values:")
print(y_prediction)

print("Actual values:")
print(y_test)

# --------------------------------------------------
# 5. Metrics evaluation for Linear Regression
# --------------------------------------------------

R2 = r2_score(y_test, y_prediction)
MSE = mean_squared_error(y_test, y_prediction)
RMSE = np.sqrt(MSE)

print("Linear Regression R2 Score:", R2)
print("Linear Regression MSE:", MSE)
print("Linear Regression RMSE:", RMSE)

# --------------------------------------------------
# Random Forest Regression
# --------------------------------------------------

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Random Forest predicted values:")
print(y_pred)

# --------------------------------------------------
# Metrics evaluation for Random Forest
# --------------------------------------------------

R2_Random = r2_score(y_test, y_pred)
MSE_Random = mean_squared_error(y_test, y_pred)
RMSE_Random = np.sqrt(MSE_Random)

print("Random Forest R2 Score:", R2_Random)
print("Random Forest MSE:", MSE_Random)
print("Random Forest RMSE:", RMSE_Random)

# Compare the models
print("\nModel Comparison")
print("Linear Regression R2:", R2)
print("Random Forest R2:", R2_Random)
print("Linear Regression RMSE:", RMSE)
print("Random Forest RMSE:", RMSE_Random)

