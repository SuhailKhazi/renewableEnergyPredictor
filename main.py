#%%
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import time
from mlxtend.evaluate import bias_variance_decomp
# %%
# datasets received from https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
#in this section we analyze the datasets and observe their shape
df_energy_dataset = pd.read_csv('energy_dataset.csv')
print(df_energy_dataset.shape)
df_weather_features = pd.read_csv('weather_features.csv')
print(df_weather_features.shape)
#%%
# PREPROCESSING OF THE WEATHER DATASET
# change the column name to time so that we can do a merge of the datasets based on this column
df_weather_features.insert(loc=0, column='time', value=df_weather_features['dt_iso'])

# drop the duplicates based on the timestamp and city name, so that for each individual timestamp, we have 5 rows because of 5 cities.
df_energy_dataset = df_energy_dataset.drop_duplicates(subset=['time'])
df_weather_features = df_weather_features.drop_duplicates(
  subset = ['time', 'city_name'],
  keep = 'last')

# here all the values are 5. this is because there is one time stamp per city. This is to ensure that no more duplicates exist
df_weather_features.time.value_counts()

# for each individual timestamp, we aggregate the data of 5 cities and take the mean, so to represent an overall weather metric for one time.
df_weather_features_aggregated = df_weather_features.groupby('time', as_index=False).mean()

# include only the relevant columns from the weather data
df_weather_features_aggregated = df_weather_features_aggregated[['time','temp','humidity','pressure','wind_speed','wind_deg','rain_3h','clouds_all']]

#%%
# PREPROCESSING OF THE ENERGY DATASET
# for this analysis, we only want to look at wind, solar and hydroelectric sources, therefore we only select those columns and get total power generated from these sources
df_energy_dataset['total_power_generated'] =  df_energy_dataset['generation hydro water reservoir'] + df_energy_dataset['generation solar'] + df_energy_dataset['generation wind onshore']

#%%
# here we prepare our output variable, the power_category.

# since we have 3 categories namely low, medium and high, we need to allocate each category into equal segments, therefore using quantiles of 33%,67% and 100%
low_range = df_energy_dataset['total_power_generated'].quantile(0.33)
medium_range = df_energy_dataset['total_power_generated'].quantile(0.67)
high_range = df_energy_dataset['total_power_generated'].quantile(1)
bins = [0,low_range, medium_range, high_range]
labels = ["low","medium","high"]
df_energy_dataset['power_category'] = pd.cut(df_energy_dataset['total_power_generated'], bins=bins, labels=labels)


# create a new dataframe from the weather dataframe and include this power_category in the last column. This df shall be used for our training and testing of model.
df_combined = df_weather_features_aggregated.copy()
df_combined['power_category'] = df_energy_dataset['power_category']
df_combined = df_combined[df_combined['power_category'].notna()]
#%%
# for our predictions, time is not a variable so can be discarded.
df_combined.drop(['time'], axis=1, inplace=True)

#%%
# now we separate the feature variables into X and the output variable into y
y = df_combined.pop('power_category')
X = df_combined
# %%

# to ensure that we are able to predict our classification algoritm, we need to encode this power_category so it predicts.
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# now we split our data into training and testing data and shall allocate 30% of this data for testing and the rest for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=20,shuffle=True,
stratify=y)
# %%
# We shall train our baseline model. For this, Naive Bayes shall be used.
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("training accuracy: ",nb.score(X_train, y_train))
print(classification_report(y_test,y_pred_nb))

cm = confusion_matrix(y_test, y_pred_nb, labels=nb.classes_)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=nb.classes_)
disp_nb.plot()


# Get Bias and Variance - bias_variance_decomp function
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
nb, X_train.values, y_train,X_test.values, y_test, 
loss='0-1_loss',
random_seed=123,
num_rounds=1000)
# Display Bias and Variance
print(f'Average Expected Loss: {round(avg_expected_loss, 4)}')
print(f'Average Bias: {round(avg_bias, 4)}')
print(f'Average Variance: {round(avg_var, 4)}')
# %%
# here we shall test our random forest classifier with its default setting and without any sort of tuning.
clf = RandomForestClassifier(random_state=20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("training accuracy: ",clf.score(X_train, y_train))
print("testing accuracy: ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp_clf = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp_clf.plot()

# Get Bias and Variance - bias_variance_decomp function
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
clf, X_train.values, y_train,X_test.values, y_test,  
loss='0-1_loss',
random_seed=123,
num_rounds=5)
# Display Bias and Variance
print(f'Average Expected Loss: {round(avg_expected_loss, 4)}')
print(f'Average Bias: {round(avg_bias, 4)}')
print(f'Average Variance: {round(avg_var, 4)}')
# %%
# now we shall do hyperparameter tuning of random forest classifier using GridSearchCV
start_time = time.time()
model = RandomForestClassifier(random_state=20)
hyperparameters = [{
'max_depth':[100,200,300,400,500],
'n_estimators':[100,200,300,400,500],
'criterion':['entropy','gini'],
'max_features':['sqrt', 'log2', 'auto'],
'min_samples_split':[1,2,3,4,5],
'min_samples_leaf':[1,2,3,4,5]
}]

optimizer = GridSearchCV(model,param_grid=hyperparameters,scoring="accuracy",cv=3,return_train_score=True)
optimizer.fit(X_train, y_train)

print("best params: ",optimizer.best_params_)
print("best score: ",optimizer.best_score_)

print("time elapsed: ",time.time() - start_time)
# %%
# running the classifier with the optimum parameters after gridsearch
clf = RandomForestClassifier(random_state=20,max_depth=500,n_estimators=300,criterion='entropy',max_features='sqrt',min_samples_split=2,min_samples_leaf=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("training accuracy: ",clf.score(X_train, y_train))
print("testing accuracy: ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp_clf = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp_clf.plot()


# # Get Bias and Variance - bias_variance_decomp function
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
clf, X_train.values, y_train,X_test.values, y_test, 
loss='0-1_loss',
random_seed=123,
num_rounds=5)
# Display Bias and Variance
print(f'Average Expected Loss: {round(avg_expected_loss, 4)}')
print(f'Average Bias: {round(avg_bias, 4)}')
print(f'Average Variance: {round(avg_var, 4)}')
# %%