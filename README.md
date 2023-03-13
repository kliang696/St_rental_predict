# U.S House Rental Prediction

The rental house market is rapidly changing, creating challenges for stakeholders to keep up with market trends and fluctuations. Accurately predicting rental house prices has become critical for success in a competitive market.A reliable prediction model can provide valuable insights and enable stakeholders to make informed decisions and set competitive prices to ensure that rental properties remain desirable and profitable.
<img  alt="Screen Shot 2023-03-11 at 23 04 04" src="https://user-images.githubusercontent.com/89816441/224523650-8e897103-8eb6-4fbe-9883-fe654d5180df.png" width=700 height=400>


## Data Description
The dataset used for predicting rental house prices is sourced from Craigslist, a popular online platform that lists over thousands of rental properties. The dataset includes 360K rental house  and contains critical variables such as state, region, parking options, laundry options, and most importantly, price, providing a comprehensive understanding of the rental house market. By leveraging this dataset, stakeholders can gain a deeper understanding of market trends and fluctuations, adjust their pricing strategies, and maximize revenue and profitability.

## Table of Contents
- Data Analytics & Dashboard with Tableau
- [Exploratory Data Analysis](#exploratory-data-analysis-and-feature-enginerring)
- [Modelling & Performance Evaluation](#model-performance-evaluation)
- [Feature Importance](#feature-importance)
- [Build Web App and deployment with Streamlit](#deployment)
- [Conclusion](#conclusion)

## Exploratory Data Analysis & Feature Engineering
1. __Missing Values__
```python
my_df.isnull().sum()
```
<img width="304" alt="Screen Shot 2023-03-11 at 22 42 00" src="https://user-images.githubusercontent.com/89816441/224522980-7afe6b2e-57ae-4eec-8a41-168fab49351f.png">

The missing value report reveals that there are 70,000 missing values for laundry_options and 140,000 missing values for parking_options, while only 1,918 rows have missing values for latitude and longitude. To address these issues, we will replace the missing values for laundry_options with "no laundry on site" and  "no parking" for parking_options as they are common options when neither is available. However, since there are only a small number of rows with missing values, we will drop them accordingly.
```python
my_df["parking_options"].fillna("no parking", inplace=True)
my_df["laundry_options"].fillna("no laundry on site", inplace=True)
my_df.dropna(inplace=True)
```



2. __Drop the unnessary column__
To ensure that the dataset is  optimized for modeling purposes, we will remove the irrelevant columns, such as 'url', 'id', 'region_url', 'image_url', and 'description'. These columns do not provide any information that is relevant to predicting rental house prices, and thus their exclusion will not impact the accuracy of the prediction model.
```python
df=my_df.drop(columns=['url', 'id','region_url','image_url', 'description'])
```

3. __Outlier Handling__
To ensure that the dataset contains only livable and relevant rental properties, we will exclude properties with more than five bedrooms and bathrooms. We will also ensure that the price of each rental property is greater than 0 and the property size is greater than 200 square feet.
```python
df = df[(df['beds'] <= 5) & (df['beds'] > 0)]
df = df[(df['baths'] <= 5) & (df['baths'] > 0)]
df = df[df['price']>100]
df = df[df['sqfeet']>200]
```
After analyzing the rental house price and size data, we have identified the existence of numerous outliers. 
<table><tr>
<td><img width="400" alt="Screen Shot 2023-03-11 at 22 47 15" src="https://user-images.githubusercontent.com/89816441/224523145-d2bb8f2d-abdf-464a-8523-c47f73cc7038.png">
<td><img width="400" alt="Screen Shot 2023-03-11 at 22 47 39" src="https://user-images.githubusercontent.com/89816441/224523156-29062b32-2b56-462e-ad36-049e678dfce0.png">
</tr></table>


To address this issue, we will use the percentile capping technique to limit the data to the 0-99 percentile range. Any data points exceeding the 99 percentile will be transformed to the threshold of 99. This technique will enable us to remove the outliers and ensure that the dataset is more representative of the rental house market's general trends and characteristics.
```python
lower_threshold = 0
upper_threshold = 0.01

for i in range(len(num_price_sf)):
    column = num_price_sf[i]
    df[column] = winsorize(df[column], limits=(lower_threshold, upper_threshold))
```
<table><tr>
<td><img width="407" alt="Screen Shot 2023-03-11 at 22 48 07" src="https://user-images.githubusercontent.com/89816441/224523178-17563e6b-2a65-4c58-aad2-126078ab0bee.png">
<td><img width="407" alt="Screen Shot 2023-03-11 at 22 48 20" src="https://user-images.githubusercontent.com/89816441/224523186-67a2cc76-47d8-47b8-ad1d-0dee08affe3b.png">
</tr></table>



4. __Duplicated Row__
Upon examining the rental house dataset, we have determined that there are no duplicate rows present in the dataset. This means that each row in the dataset is unique and does not contain any exact copies of other rows.
```python
duplicates = my_df.duplicated()
print("Number of duplicates:", duplicates.sum())
```
5. __Heat Map__
The heatmap will display the correlation between different variables, with the strength of the correlation indicated by the intensity of the color. By analyzing the heatmap, we can identify the variables that have the strongest correlation with rental prices and use this information to make informed decisions and set competitive prices for rental properties.
<img width="800" alt="Screen Shot 2023-03-11 at 22 54 37" src="https://user-images.githubusercontent.com/89816441/224523381-77e89da4-1608-4e0a-b3e7-0e212c639959.png">
Upon analyzing the heatmap, we have found that the top three features that are most correlated with rental prices are sqfeet, beds, and baths. This finding aligns with our expectations, as we would anticipate that as the property size and number of bedrooms and bathrooms increase, the rental price would also increase accordingly.


6. __Categorical Feature Encoding__
Since the categorical values in the dataset are nominal, we will use the pandas `get_dummies()` function instead of the label encoder to create new columns based on each category. By using `get_dummies()`, we can create new columns for each category, such as state_ca, state_ny, etc.
```python
cat= ['region', 'type', 'laundry_options', 'parking_options', 'state']
df_encoded = pd.get_dummies(df, columns=cat)
```
## Modelling Evaluation & Improvement
The main metrics used to evaluate the model performance were Mae, Mse, Rmse, and R2. The table below clearly indicates that Catboost performed  better than both Lgbm and Linear regression. After Hyper-parameter tunning, Lgbm outperformed the catboost and linear regression.
<tr> - Before Hyper-parameter tunning:
<table><tr>
<tr><img width="539" alt="Screen Shot 2023-03-11 at 23 41 21" src="https://user-images.githubusercontent.com/89816441/224524788-4e1df892-baf8-4b80-856a-ea64181d24d7.png">
<tr> - After Hyper-parameter tunning:
<tr><img width="539" alt="Screen Shot 2023-03-11 at 23 41 52" src="https://user-images.githubusercontent.com/89816441/224524808-5d88b16d-b465-4f89-8859-44f46e8b83d2.png">
</tr></table>

## Feature Importance

<img width="585" alt="Screen Shot 2023-03-12 at 22 16 58" src="https://user-images.githubusercontent.com/89816441/224593008-711cb0f8-4e73-41a8-a3b3-13e43ab2ba41.png">






    
