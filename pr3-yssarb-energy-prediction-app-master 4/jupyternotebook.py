#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'data'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install xgboost')
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

#%% [markdown]
# Data

#%%
energy = pd.read_csv('../Energy_Project/Resources/AEP_hourly.csv', index_col=[0], parse_dates=[0])


#%%
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = energy.plot(style='.', figsize=(15,5), color=color_pal[0], title='AEP')

#%% [markdown]
# Train/Test Split

#%%
split_date = '01-Oct-2014'
energy_train = energy.loc[energy.index <= split_date].copy()
energy_test = energy.loc[energy.index > split_date].copy()


#%%
_ = energy_test     .rename(columns={'AEP_MW': 'TEST SET'})     .join(energy_train.rename(columns={'AEP_MW': 'TRAINING SET'}), how='outer')     .plot(figsize=(15,5), title='AEP', style='.')

#%% [markdown]
# Create Time Series Features

#%%
def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


#%%
X_train, y_train = create_features(energy_train, label='AEP_MW')
X_test, y_test = create_features(energy_test, label='AEP_MW')

#%% [markdown]
# Create XGBoost Model

#%%
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) # Change verbose to True if you want to see it train

#%% [markdown]
# Feature Importances
# Feature importance is a great way to get a general idea about which features the model is relying on most to make the prediction. This is a metric that simply sums up how many times each feature is split on.
# 
# We can see that the day of year was most commonly used to split trees, while hour and year came in next. Quarter has low importance due to the fact that it could be created by different dayofyear splits.

#%%
_ = plot_importance(reg, height=0.9)

#%% [markdown]
# Forecast on Test Set

#%%
energy_test['MW_Prediction'] = reg.predict(X_test)
energy_all = pd.concat([energy_test, energy_train], sort=False)


#%%
_ = energy_all[['AEP_MW','MW_Prediction']].plot(figsize=(15, 5))

#%% [markdown]
# Look at the First Month of Predictions

#%%
# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = energy_all[['MW_Prediction','AEP_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='10-01-2014', upper='11-01-2014')
ax.set_ylim(0, 60000)
plot = plt.suptitle('October 2014 Forecast vs Actuals')


#%%
# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = energy_all[['MW_Prediction','AEP_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='10-01-2014', upper='10-08-2014')
ax.set_ylim(0, 60000)
plot = plt.suptitle('First Week of October Forecast vs Actuals')


#%%
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = energy_all[['MW_Prediction','AEP_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='07-01-2015', upper='07-08-2015')
plot = plt.suptitle('First Week of July Forecast vs Actuals')

#%% [markdown]
# Error Metrics on Test
#%% [markdown]
# Our RMSE error is 2636882/
# Our MAE error is 1295.63/
# Our MAPE error is 8.8%

#%%
mean_squared_error(y_true=energy_test['AEP_MW'],
                   y_pred=energy_test['MW_Prediction'])


#%%
mean_absolute_error(y_true=energy_test['AEP_MW'],
                   y_pred=energy_test['MW_Prediction'])

#%% [markdown]
# We are using mean absolute percent error because it gives an easy to interperate percentage showing how off the predictions are. MAPE isn't included in sklearn so we need to use a custom function.

#%%
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


#%%
mean_absolute_percentage_error(y_true=energy_test['AEP_MW'],
                   y_pred=energy_test['MW_Prediction'])

#%% [markdown]
# Look at Worst and best Predicted Days

#%%
energy_test['error'] = energy_test['AEP_MW'] - energy_test['MW_Prediction']
energy_test['abs_error'] = energy_test['error'].apply(np.abs)
error_by_day = energy_test.groupby(['year','month','dayofmonth'])     .mean()[['AEP_MW','MW_Prediction','error','abs_error']]


#%%
# Over forecasted days
error_by_day.sort_values('error', ascending=True).head(10)

#%% [markdown]
# Interesting Findings: The top dates are holidays; it would be cool to chart weather data against this to see if there was a cold front at the end of February

#%%
# Worst absolute predicted days
error_by_day.sort_values('abs_error', ascending=False).head(10)

#%% [markdown]
# The best predicted days seem in October

#%%
# Best predicted days
error_by_day.sort_values('abs_error', ascending=True).head(10)

#%% [markdown]
# Plotting some best/worst predicted days

#%%
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = energy_all[['MW_Prediction','AEP_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 35000)
ax.set_xbound(lower='12-25-2015', upper='12-26-2015')
plot = plt.suptitle('Dec 25, 2015 - Worst Predicted Day')

#%% [markdown]
# It was a high of 8 degrees in West Virginia on 02/19/2015, well below the average daily temperature of 34, and a low of 1 degree! In short, it was colder than normal which may have resulted in the model's undestimation.

#%%
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = energy_all[['MW_Prediction','AEP_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='03-13-2015', upper='03-14-2015')
plot = plt.suptitle('March 13, 2015 - Best Predicted Day')

#%% [markdown]
# This one is nearly spot on!
#%% [markdown]
# Things to investigate: Weather Data, Holiday Indicators, Local Economic Activity Analysis

