# Melbourne Housing Price Prediction

## Project Overview
This project aims to analyze and predict housing prices in Melbourne using machine learning models. The dataset consists of property sales records, including features such as location, number of rooms, type of property, and various other attributes. The primary goal is to build predictive models that estimate property prices based on these features.

## Files and Directories
- **models/**
  - `RandomForest_model.pkl` - Trained Random Forest model.
  - `XGBoost_model.pkl` - Trained XGBoost model.
- **data/**
  - `cleaned_dataset.csv` - Processed dataset after cleaning and feature engineering.
  - `melb_data.csv` - Raw dataset containing Melbourne housing records.

## Dependencies
The following Python libraries are required for data processing, visualization, and model training:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.linear_model import LinearRegression    
from xgboost import XGBRegressor                    
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import pickle
```

## Data Preprocessing
### Loading Data
```python
melb_data = pd.read_csv('data/melb_data.csv')
```
### Data Inspection
```python
melb_data.info()
melb_data.head()
```
### Data Cleaning
- Handling missing values
- Removing duplicates
- Converting data types
- Encoding categorical variables
- Feature engineering (e.g., extracting location details using Google Maps API)

```python
melb_data = melb_data.drop_duplicates()
melb_data = pd.get_dummies(melb_data, columns=['Type'])
melb_data['Date'] = pd.to_datetime(melb_data['Date'], dayfirst=True)
melb_data.drop(['Method', 'SellerG'], axis=1, inplace=True)
```

## Google Maps API Integration
To retrieve missing geographical coordinates and council areas, we use Google Maps API.

```python
def get_coordinates_from_address(address, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if results.get('status') == 'OK':
            location = results['results'][0]['geometry']['location']
            return location['lat'], location['lng']
    return None, None
```

## Exploratory Data Analysis
- Checking distributions of numerical and categorical features
- Identifying correlations between variables
- Handling outliers

```python
plt.figure(figsize=(10,8))
sns.heatmap(melb_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```

## Model Training
### Baseline Model
A simple XGBoost regression model is used as a baseline.

```python
X = melb_data.drop('Price', axis=1)
y = melb_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

### Model Evaluation & Selection
We train multiple models and compare their performance using cross-validation.

```python
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} CV MSE: {-np.mean(scores):.4f}")
```

### Saving the Models
```python
for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f'models/{name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

print("Models have been saved successfully.")
```

## Results
- **Random Forest Model:** CV MSE ~ 83.7 billion
- **XGBoost Model:** CV MSE ~ 79.7 billion

## Future Improvements
- Hyperparameter tuning for better model performance
- Feature selection to reduce dimensionality
- Using additional datasets to improve prediction accuracy

## Usage
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run data preprocessing and analysis:
   ```sh
   python preprocess.py
   ```
3. Train and evaluate models:
   ```sh
   python train.py
   ```
4. Load trained models for predictions:
   ```python
   with open('models/XGBoost_model.pkl', 'rb') as file:
       model = pickle.load(file)
   predictions = model.predict(X_test)
   ```

## Contact
For any inquiries, please reach out via email or GitHub.

---
This README provides a complete guide on how the Melbourne Housing Price Prediction project is structured, the preprocessing steps taken, model training, and future improvements.

