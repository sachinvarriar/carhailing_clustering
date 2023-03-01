# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:44:36 2023

@author: sachin
"""


import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Importing both tables


car_trips = pd.read_csv(r'C:\Users\sachi\car_ride_sim\sim_n_cluster\cab_hiring_details.csv')
user = pd.read_csv(r'C:\Users\sachi\car_ride_sim\sim_n_cluster\user.csv')


# Viewing contents of the car_trip table.
# 
# Note: This data was simulated with the help of some new york taxi data from the year 2016 
# available in sample data of Google bigquery. The intend behind using this data set was to 
# get correct coordinates within a particular city and the actual distance by car between 
# those two coordinates according to the google maps coordinates. 
# I did try to use the google maps api, which would have given data as an api response that 
# can be stored as a txt file. However google api would have only a limited number of free 
# hits. The open map api was not responding at the time of creating this notebook, 
# as a result of which I had to go with the pre-populated map coordinates for new york taxi zones from bigquery. 


car_trips.head()


# The user table was simulated using the RANDBETWEEN function excel. There are a total of 500 users. 
#In this data the Gender variable is also randomly simulated. 0 = female, 1 = male.


user.head()


# While creating the data, the user_id was created in different formats. 
#The code below is a few preprocessing steps to join the two tables and also preparing columns for group by/pivotting.



#preparing the user_id column for joining
car_trips['user_id'] = car_trips['user_id']+1000000

car_trips['user_id1'] = car_trips['user_id'].astype('str')
car_trips['user_id']= 'u'+ car_trips['user_id1'].str.slice(1,7)

user['age_group'] = pd.cut(user['age'],6, precision=0)

user['Gender']=user['Gender'].apply(lambda x: 'Male' if x == 1 else 'Female')

#taking a join on the user_id
trip = car_trips.merge(user, on = 'user_id')

trip = trip.drop(columns = ['user_id1'])

#calculating the duration of each ride
trip['end_time']=pd.to_datetime(trip['end_time'], format='%Y-%m-%d %H:%M:%S.%f %Z', errors='coerce')
trip['start_time']=pd.to_datetime(trip['start_time'], format='%Y-%m-%d %H:%M:%S.%f %Z', errors='coerce')
trip['duration'] = trip['end_time']-trip['start_time'] 

trip['dur']=trip['duration'].astype('timedelta64[m]')


trip.head()


trip = trip[trip['trip_distance']>0]
trip = trip[trip['dur']>0]


# Task:
# 
#     Using pandas and appropriate python libraries perform the statistical analysis for each gender and age group and compute
# 
#     Average distance of trips.
#     Total number of trips taken.
#     Total distance covered.
#     Total time spent on trips.



trip_d_age_g=trip[["age_group", "trip_distance"]]



trip_d_age_g.groupby('age_group').pipe(lambda x: x.mean())


trip_d_age_g.groupby('age_group').pipe(lambda x: x.max())

trip[trip['trip_distance']>30]
# there seems to be one outlier. We shall remove this outlier as it may effect our clustering


trip=trip[trip['trip_distance']<=30.67]


# #### Average distance of trips by age group


trip_d_age_g=trip[["age_group", "trip_distance"]]
trip_d_age_g.groupby('age_group').pipe(lambda x: x.mean())


# #### No of trips by age group



trip_d_age_g=trip[["age_group", "trip_id"]]
trip_d_age_g.groupby('age_group').pipe(lambda x: x.count())


# #### No of users by age group



user_age_g=user[["age_group", "user_id"]]
user_age_g.groupby('age_group').pipe(lambda x: x.count())


# #### Total distance travelled by age group



trip_d_age_g=trip[["age_group", "trip_distance"]]
trip_d_age_g.groupby('age_group').pipe(lambda x: x.sum())


# #### Total time spent on trips in minutes



trip_t_age_g=trip[["age_group", "duration"]]
trip_t_age_g.groupby('age_group').pipe(lambda x: x.sum())


# #### Average distance of trips by gender


trip_d_g=trip[["Gender", "trip_distance"]]
trip_d_g.groupby('Gender').pipe(lambda x: x.mean())


# #### No of trips by Gender



trip_n_g=trip[["Gender", "trip_id"]]
trip_n_g.groupby('Gender').pipe(lambda x: x.count())


# #### Total distance travelled by age group


trip_d_g.groupby('Gender').pipe(lambda x: x.sum())


# Total duration by Gender



trip_t_g=trip[["Gender", "duration"]]
trip_t_g.groupby('Gender').pipe(lambda x: x.sum())



trip_agg1=trip[["user_id", "dur", "trip_distance"]]
trip_agg_1=trip_agg1.groupby('user_id').pipe(lambda x: x.sum())




trip_agg2=trip[["user_id", "dur", "trip_distance"]]
trip_agg_2=trip_agg2.groupby('user_id').pipe(lambda x: x.mean())




user.columns



user_to_merge = user.drop(columns = ['age_group'])


user_to_merge.head()




trip_agg3=trip_agg_1.merge(trip_agg_2, on = 'user_id')




trip_agg=trip_agg3.merge(user_to_merge, on = 'user_id')



trip_agg.head()




trip_agg.columns = ['user_id','duration_total','distance_total','duration_average','distance_average','age','Gender']




user_agg=trip_agg.drop(columns=['user_id'])
user_agg


from sklearn.decomposition import PCA
pca = PCA(n_components=3, whiten=True)
Num_features=user_agg.select_dtypes(include=[np.number]).columns
x=user_agg[Num_features]
principalComponents = pca.fit_transform(x)

# Cumulative Explained Variance
cum_explained_var = []
for i in range(0, len(pca.explained_variance_ratio_)):
    if i == 0:
        cum_explained_var.append(pca.explained_variance_ratio_[i])
    else:
        cum_explained_var.append(pca.explained_variance_ratio_[i] + 
                                 cum_explained_var[i-1])

print(cum_explained_var)





#Principal Components converted to a Data frame
principalDf  = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
principalDf.shape




#Concatenating the PCAs with the categorical variable
finalDf_Cat = pd.concat([principalDf, user_agg['Gender']], axis = 1)
finalDf_Cat.head(2)



from kmodes.kprototypes import KPrototypes
cost = []
X = finalDf_Cat
for num_clusters in list(range(2,7)):
    kproto = KPrototypes(n_clusters=num_clusters, init='Huang', random_state=42,n_jobs=-2,max_iter=15,n_init=50) 
    kproto.fit_predict(finalDf_Cat, categorical=[3])
    cost.append(kproto.cost_)

plt.plot(cost)
plt.xlabel('K')
plt.ylabel('cost')
plt.show





# Converting the dataset into matrix
X = finalDf_Cat.to_numpy()




# Running K-Prototype clustering
kproto = KPrototypes(n_clusters=2, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=.25) 
clusters = kproto.fit_predict(X, categorical=[3])

import seaborn as sns
#Visualize K-Prototype clustering on the PCA projected Data
df=pd.DataFrame(finalDf_Cat)
df['Cluster_id']=clusters
print(df['Cluster_id'].value_counts())
sns.pairplot(df,hue='Cluster_id',palette='Dark2',diag_kind='kde')


# Converting the dataset into matrix
X = user_agg.to_numpy()
# Running K-Prototype clustering
kproto = KPrototypes(n_clusters=3, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=0.15) 
clusters = kproto.fit_predict(X, categorical=[5])
#Visualize K-Prototype clustering
df=pd.DataFrame(user_agg)
df['Cluster_id']=clusters
print(df['Cluster_id'].value_counts())
sns.pairplot(df,hue='Cluster_id',palette='Dark2',diag_kind='kde')

#We tried clustering using pca and with the variables as such. The pca components 1 covers 99% of the data,
# hence the result of clustering with the actual dataset would be better

#We can now use this data for supervised classification

df['Gender'].replace(['Female', 'Male'],
                        [0, 1], inplace=True)
X=df[['duration_total', 'distance_total', 'duration_average','distance_average', 'age', 'Gender']]
y = df['Cluster_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # 75% training and 25% validation



# fuction to create models for tuning
def build_models():
    
#     dic of models
    Ada_models = dict()
    
    # number of decision stumps
    decision_stump= [4, 8, 16, 20, 25,30,35 ,50]
        
#    using for loop to iterate though trees
    for i in decision_stump:
        for j in np.arange(0.1, 2.1, 0.1):
            Ada_models[str(i),str(j)] = AdaBoostClassifier(n_estimators=i, learning_rate=j)
    return Ada_models

# function for the validation of model
def evaluate_model(model, Input, Ouput):
    
    # defining the method of validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    
    
    # validating the model based on the accurasy score
    accuracy = cross_val_score(model, Input, Ouput, scoring='accuracy', cv=cv, n_jobs=-1)
    
#      accuracy score- hyperparameter tuning of Adaboost
    return accuracy

# calling the build_models function
models = build_models()
# creating list
results, names = list(), list()
# using for loop to iterate thoug the models
for name, model in models.items():
    
    # calling the validation function
    scores = evaluate_model(model, X_train, y_train)
    
    
    # appending the accuray socres in results
    results.append(scores)
    names.append(name)
    
    
    # printing results of hyperparameter tuning of Adaboost
    print('---->Stump tree (%s)---Accuracy( %.5f)' % (name, mean(scores)))

#We see that 16 stumps with learning rate of 0.1 gives best accuracy
    
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=16,
                             learning_rate=0.1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
