---
jupyter:
  jupytext:
    notebook_metadata_filter: hide_input
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Car accident severity predictor

This is the project report by Katharina Egert, submitted October 2020.

## 1. Introduction

The objective of this project is to create a way to predict severity of car accidents happening based on road and weather conditions.
That is, our algorithm needs to take as input the conditions that may impact severity of accidents:

* road conditions
* weather conditions
* lighting conditions

etc.

and predict the risk profile of potential accidents, i.e. the severity label. A user should then be able to specify the current conditions
of their itinerary and should then get back the severity label of accidents happening.

Example: A wet road should potential lead to higher severity accidents than a dry road.

The main stakeholders are:

* __traffic regulators__: They can make policies (e.g. speed limits, higher controls etc) in order to mitigate risks.
* __drivers__: They can recognize risks in advance and adapt their behavior in order to lower their own personal risks, e.g. drive more carefully or even avoid travelling at at risk conditions all together.


```python tags=["hide_input"]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore') # just for the sake of beauty

```


## 2. Data 

In this section we will descibe the underlying data.

### 2.1 Data Source

The given data set contains all collisions provided by Seattle police departement and recorded by Traffic Records. The level of aggregation is
weekly. The timeframe is 2004 to today.
(See metadata from the SPD Collision data set.)

### 2.2 Data Loading
In the next step, we load the data into the notebook and display the first 5 lines of the raw table. Not all columns will be used and some will need to be transformed in order to be exploitable for our later classification.

```python tags=["hide_input"]
full_df = pd.read_csv("../Collisions.csv", low_memory=False)
print(full_df.columns)
full_df.head()
```

<!-- #region -->
### 2.3 Data Description

The problem consists in predicting accident severity based on outer conditions of accidents recorded. 
The label to be predicted is 'SEVERITYCODE'. A description of the severity is contained in the column 'SEVERITYDESC' and 'SEVERITYCODE.1'
which is therefore redundant.



The following columns are of a technical nature and therefore of no use to us:

* 'OBJECTID',
* 'INCKEY', 
* 'COLDETKEY', 
* 'REPORTNO',
* 'STATUS'
* 'INTKEY'
* 'EXCEPTRSNCODE',
* 'EXCEPTRSNDESC'

There are several other fields in the data which might be good predictors for severity, however it does not make sense to use these for this problem as traffic planers as well as drivers will not be able to input this data when using this analysis. Example: for pedestrian/cyclist count (which is actually depending on the dependent variable!) as one will not know how many pedestrians/bikes will be on the road on a given day. 

The total list of colums concerned is:
* 'PERSONCOUNT', 
* 'PEDCOUNT', 
* 'PEDCYLCOUNT', 
* 'VEHCOUNT',
* 'INATTENTIONIND',
* 'HITPARKEDCAR',
* 'HITPARKEDCAR',
* 'PEDROWNOTGRNT',
* 'SDOTCOLNUM', 
* 'ST_COLCODE', 
* 'ST_COLDESC',
* 'SEGLANEKEY', 
* 'CROSSWALKKEY',
* 'SDOT_COLCODE'
* 'SDOT_COLDESC',
* 'COLLISIONTYPE',

The colums that can be explored are
* 'ADDRTYPE', 'X', 'Y'
* 'LOCATION',
* 'INCDATE' and 'INCDTTM',
* 'JUNCTIONTYPE',
* 'WEATHER'
* 'ROADCOND', 
* 'LIGHTCOND'
* 'UNDERINFL'
* 'SPEEDING'
       
  


<!-- #endregion -->

<!-- #region -->
# 3. Methodology
This section represents the main component of the report where I will discuss and describe exploratory data analysis and inferential statistical testing performed.


### 3.1 Label analysis
We need to predict severity of the accident given by the data colum 'SEVERITYCODE'. The following codes are given:

* 3—fatality 
* 2b—serious injury
* 2—injury
* 1—property damage
* 0—unknown



We will now analyze the frequency of their occurrences. The first static gives the absolute frequency, the second one the relative frequency.
<!-- #endregion -->

```python tags=["hide_input"]
# first preprocessing: select columns to reduce size
df = full_df[['SEVERITYCODE', 'ADDRTYPE', 'LOCATION','INCDATE', 'INCDTTM','JUNCTIONTYPE','WEATHER','ROADCOND', 'LIGHTCOND','UNDERINFL','SPEEDING']]
# drop missing and unknown severity codes
df = df.dropna(subset=['SEVERITYCODE'])
df = df[df['SEVERITYCODE']!= "0"] 
```

We check now the total frequency of occurrences.

```python tags=["hide_input"] hide_input=true
df['SEVERITYCODE'].value_counts()
```

```python tags=["hide_input"]
explode = (0, 0, 0.2, 0.3)  # only "explode" the 2nd slice (i.e. 'Hogs')
labels = df['SEVERITYCODE'].unique().tolist()
values = df['SEVERITYCODE'].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
df['SEVERITYCODE'].value_counts().apply(lambda r: r/df['SEVERITYCODE'].count())
```

<!-- #region -->
Overall, light accidents dominate by nearly 2/3 and the remainder is mostly light injuries. Heavy injuries consist only of 1.5% and fatalities about 0.2%, thus much more limited and thus much less significant data. 

This means that out data set is unbalanced, hence we need to try later on to account for this imbalance.

Secondly, for the algorithm to make sense from the stake holder point of view, it is much more important to detect the risk of potential injury-type accidents than misclassifiying an actual property damage case, since one is rather too careful than take too much risk. This will help us shape the cost function.


###  3.2 Feature selection

In this section, we will select the features which determine the features to use. For this we will first use business knowledge to select obvious factors and then also check which column seems to have the most impact on the severity outcome. 

* 'WEATHER' containing data on the weather such at if it was dry or wet etc.
* 'ROADCOND' containing information whether the road was dry, wet etc.
* 'LIGHTCOND' containing data on lighting, e.g. if it was dark.

We will check for these three first and eliminate unusable columns.

<!--

 'INATTENTIONIND', 'UNDERINFL', 'SPEEDING',

Full colums
SEVERITYCODE', 'X', 'Y', 'OBJECTID', 'INCKEY', 'COLDETKEY', 'REPORTNO',
       'STATUS', 'ADDRTYPE', 'INTKEY', 'LOCATION', 'EXCEPTRSNCODE',
       'EXCEPTRSNDESC', 'SEVERITYCODE.1', 'SEVERITYDESC', 'COLLISIONTYPE',
       'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INCDATE',
       'INCDTTM', 'JUNCTIONTYPE', 'SDOT_COLCODE', 'SDOT_COLDESC',
       'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
       'PEDROWNOTGRNT', 'SDOTCOLNUM', 'SPEEDING', 'ST_COLCODE', 'ST_COLDESC',
       'SEGLANEKEY', 'CROSSWALKKEY', 'HITPARKEDCAR'-->
       
#### 'WEATHER'

We first check for the values.
<!-- #endregion -->

```python tags=["hide_input"]
df['WEATHER'].value_counts()
```

Now we're interested to see which of these conditions correlate with higher rates of severeness of accidents. Let's first have a look at absolute occurences and then relative frequencies.

```python tags=["hide_input"]
print("Absolute Frequencies:")
#print(pd.crosstab(full_df['WEATHER'],full_df['SEVERITYCODE']))
cross = pd.crosstab(df['WEATHER'],df['SEVERITYCODE']).apply(lambda r: np.round(r/r.sum(),decimals=3), axis=1)
cross
```

```python tags=["hide_input"]
# Some technical function to plot cross tabs later
def plot_cross(cross):
    """ Plots a cross matrix"""
    fig = plt.figure(figsize=[16,9])
    ax1 = fig.add_subplot(221)

    ax1.set_title("1 - Property Damage")

    ax2 = fig.add_subplot(222)
    ax2.set_title("1 - Light Injuries")

    ax3 = fig.add_subplot(223)
    ax3.set_title("2b - Serious Injuries")

    ax4 = fig.add_subplot(224)
    ax4.set_title("3 - Fatalities")

    cross['1'].plot(ax=ax1, color = ['blue'], kind='bar')
    cross['2'].plot(ax=ax2, color = ['orange'], kind='bar')
    cross['2b'].plot(ax=ax3, color = ['red'], kind='bar')
    cross['3'].plot(ax=ax4, color = ['black'], kind='bar') 
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    fig.subplots_adjust(hspace=0.8)
```

```python tags=["hide_input"]
plot_cross(cross)
```

We see again heavy imbalanced-ness of the data. We can mitigate this by using out domain knowledge to group the data appropriately.

```python tags=["hide_input"]
weather_dict = {'Clear': 'mild','Overcast':'mild','Partly Cloudy':'mild', \
                'Snowing' : 'bad', \
                'Sleet/Hail/Freezing Rain': 'bad', \
                'Blowing Sand/Dirt': 'bad',  'Severe Crosswind': 'bad', \
                'Raining': 'raining', \
                'Unknown': 'unknown',  'Other': 'unknown'               
               }
df['WEATHER'] = df['WEATHER'].map(weather_dict)
#print("Relative Frequencies by severity:")
cross = pd.crosstab(df['WEATHER'],df['SEVERITYCODE'])#.apply(lambda r: np.round(r/r.sum(),decimals=3), axis=1)
```

We check the frequencies now visually.


```python tags=["hide_input"]
cross = pd.crosstab(df['WEATHER'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
plot_cross(cross)
```

__Observations:__

1. Most accidents happen at clear, overcast weather ('mild') or when it's raining. This is no surprise as these are the most common weather situations.
1. Most fatalities and heavy injuries also happen at mild weather or when it's raining.
1. Bad conditions such as snowing, while intuitively making up for a big risk, has actually a lower risk for severity. Perhaps drivers are already taking this risk into account and drive more carefully.
1. The most fatal weather is partly cloudy, having however not much data.

__Key takeaway__

Severe accidents happen most when it's raining or mild. Really adverse conditions are more associated with property damage or, a bit higher than for other conditions, mild injuries.



### 'ROADCOND'

We do the exact same thing for road conditions. We'll start with frequency.

```python tags=["hide_input"]
df['ROADCOND'].value_counts()
```

We note that, again, the main categories 'Dry' and 'Wet' dominate a lot the dataset. The two last categories have hardly enough data points. We check now for absolute and relative frequencies per severity code.

```python tags=["hide_input"]
print("Absolute frequency:")
pd.crosstab(df['ROADCOND'],df['SEVERITYCODE'])
```

```python tags=["hide_input"]
print("Relative frequency:")
cross = pd.crosstab(df['ROADCOND'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
cross
```

```python tags=["hide_input"]
plot_cross(cross)
```

__Observations__

1. Wet and dry are again the most fatal categories, also in relative terms.
1. Adverse conditions seem to again favor property damage only accidents.
1. We can drop again unknown and other categories.

```python tags=["hide_input"]
df = df[ (df['ROADCOND']!= "Unknown") & (df['ROADCOND']!= "Other")]
df = df.dropna(subset=['ROADCOND'])
```

After elimination of these columns, we have the following average severities.

```python tags=["hide_input"]
df['SEVERITYCODE'].value_counts().apply(lambda r: r/df['SEVERITYCODE'].count())
```

### 'LIGHTCOND'

We do the exact same thing for light conditions. We'll start with frequency.

```python tags=["hide_input"]
df['LIGHTCOND'].value_counts()
```

```python tags=["hide_input"]
df.dropna(subset=['LIGHTCOND'], inplace=True)
```

```python tags=["hide_input"]
cross = pd.crosstab(df['LIGHTCOND'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
cross
```

__Observations:__

1. Our domain knowledge tells us, the darker the more risky.
1. Most accidents happen at daylight, so lighting does not fully explain higher severity accidents.
1. Most severe accidents happen in the dark.
1. Domain: Dusk/dawn are similar, also no streetlights when dark, so we can group these together.

We drop again 'Unknown', 'Other', 'Dark - Unknown Lighting' and missing values.


```python tags=["hide_input"]
df = df[ (df['LIGHTCOND']!= "Unknown") & (df['LIGHTCOND']!= "Other") & (df['LIGHTCOND'] != "Dark - Unknown Lighting")]
df = df.dropna(subset=['LIGHTCOND'])

```

```python tags=["hide_input"]
light_dict = {'Dark - No Street Lights' : 'dark - no lights', 'Dark - Street Lights Off': 'dark - no lights', \
              'Dark - Street Lights On' : 'dark with lights', \
              'Dawn': 'dusk/dawn', 'Dusk': 'dusk/dawn', \
              'Daylight': 'daylight'\
               }
df['LIGHTCOND'] = df['LIGHTCOND'].map(light_dict)

df['LIGHTCOND'].value_counts()

```

```python tags=["hide_input"]
cross = pd.crosstab(df['LIGHTCOND'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
cross
```

```python tags=["hide_input"]
plot_cross(cross)
```

__Key observations:__ 

1. The most fatal conditions are dark nights with lighting followed by dusk and dawn. 
1. Most safest to drive is daylight follwed by dark with no lights, where there are probably less people on the route. 

### Time dimension

We do get a timing dimension via 'INCDTTM'. First, we need to cast this column as datetime.

```python tags=["hide_input"]
df['INCDTTM'].head()
df['INCDTTM'] = pd.to_datetime(df['INCDTTM'])
df['INCDTTM'].head()
```

Next, we extract dates (for our time series), extract the hour as well as the weekday.

```python tags=["hide_input"]
df['date'] = df['INCDTTM'].dt.date
df['yearmonth'] = str(df['INCDTTM'].dt.year) + "-" + str(df['INCDTTM'].dt.month)
df['hour'] = df['INCDTTM'].dt.hour
df['weekdaynum'] = df['INCDTTM'].dt.weekday

dict_weekday = {0 : '1 - Monday', 1: '2 - Tuesday', \
                2 : '3 - Wednesday',  3 : '4 - Thursday',\
                4 : '5 - Friday',  5 : '6 - Saturday', 6 : '7 - Sunday'
               }
df['weekday'] = df['weekdaynum'].map(dict_weekday)
```

#### Dates


```python tags=["hide_input"]
df['count1'] = df['SEVERITYCODE'].apply(lambda x: 1 if (x == '1') else 0)
df['count2'] = df['SEVERITYCODE'].apply(lambda x: 1 if (x == '2') else 0)
df['count2b'] = df['SEVERITYCODE'].apply(lambda x: 1 if (x == '2b') else 0)
df['count3'] = df['SEVERITYCODE'].apply(lambda x: 1 if (x == '3') else 0)

ts = df[['date','count1','count2','count2b','count3']].groupby(['date']).sum()
ts.plot(figsize=[20,5],color=['blue','orange','red','black'])



```

__Observations__
1. We see a difference between years. Therefore it might make sense to introduce a per year variable.  
1. Additionally, there seems to be a by month seasonality and potentially a per week seasonality.

```python tags=["hide_input"]
df['year'] = df['INCDTTM'].dt.year
df['month'] = df['INCDTTM'].dt.month
```

#### Years

We take a look at relative decomposition of accidents.


```python tags=["hide_input"]
cross = pd.crosstab(df['year'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
plot_cross(cross)


```

```python tags=["hide_input"]
pd.crosstab(df['year'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
```

We sse that fatality rates vary with a factor two, while the other categories fluctuate, too, but a bit less. There is no clear linear trend, so we will need to use this variable categorically.

#### Month

```python tags=["hide_input"]
cross = pd.crosstab(df['month'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
cross
```

```python tags=["hide_input"]
plot_cross(cross)
```

__Observations__

* Here we see a seasonality trend for winter and summer for property damage and light injuries, less so for serious injuries and fatalities.
* Fatalities grow at the end of year.
* There is no linear relationship, so we cannot use a numerical variable.

```python tags=["hide_input"]
df['month'] = df['month'].astype("|S")
```

#### Hour

```python tags=["hide_input"]
cross = pd.crosstab(df['hour'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
plot_cross(cross)
```

__Observations:__

1. During the day less serious accidents happen. 
1. At night, the peak hour for sever accidents is 1am.
1. The safest timeframe is in the morning.
1. The relationship is not linear, hence we cannot use it as a standard numerical variable.

```python tags=["hide_input"]
df['hour'] = df['hour'].astype("|S")
```

#### Weekday

```python tags=["hide_input"]
cross = pd.crosstab(df['weekday'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)#.plot(kind='bar')
plot_cross(cross)
```

__Obervations:__

1. Tuesday is safest for fatalities, while Wednesdays and Sundays pose the highest risk.
1. All other weekdays are pretty much equal.




### Location type

As a driver, one knows that accidents are more likely at intersections. This information is stored in the variable 'ADDRTYPE'. 

```python tags=["hide_input"]
df['ADDRTYPE'].value_counts()
```

```python tags=["hide_input"]
cross = pd.crosstab(df['ADDRTYPE'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
plot_cross(cross)
```

<!-- #region -->
__Observations__ 

* Most deaths and serious injuries happen at intersections and blocks.
* Alleys are more prone for property damage.


### 'UNDERINFL'

We see that the data is not quite clean and we need to group it together.
<!-- #endregion -->

```python tags=["hide_input"]
df['UNDERINFL'].value_counts()

```

```python tags=["hide_input"]
df['underinfl'] = df['UNDERINFL'].apply(lambda x: 1 if ((x == '1') | (x == 'Y'))  else 0)
cross = pd.crosstab(df['underinfl'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
plot_cross(cross)

```

<!-- #region -->
__Observations__

* Not surprisingly, under influence of drugs and alcohol, serious injuries and fatalities are more likely. 
* Here traffic managers may administer more drug testing in order to make these driving conditions safer.


### 'SPEEDING'

Here, we take the same steps as for UNDERINFL.
<!-- #endregion -->

```python tags=["hide_input"]
df['SPEEDING'].value_counts()
df['speeding'] = df['SPEEDING'].apply(lambda x: 1 if (x == 'Y')  else 0)
cross = pd.crosstab(df['speeding'],df['SEVERITYCODE']).apply(lambda r: r/r.sum(), axis=1)
plot_cross(cross)

```

__Observations__

Here, the same conclusion as for under influence: Serious injuries and fatalities rise when speeding.

<!-- #region -->
#### Feature summary

We will therefore use

* WEATHER (transformed)
* ROADCOND
* LIGHTCOND (transformed)
* hour (transformed from date)
* weekday (transformed from date)
* month
* ADDRTYPE
* speeding
* underinfl


We therefore reduce the columns as follows:
<!-- #endregion -->

```python tags=["hide_input"]
df = df[['SEVERITYCODE','WEATHER','ROADCOND','LIGHTCOND','hour', 'weekday', 'ADDRTYPE','year','month','speeding','underinfl']]
df.head()
```

First we convert the categorical data into dummy variables encoding our categorical variables.

```python tags=["hide_input"]
dummy_columns = ['WEATHER','ROADCOND','LIGHTCOND','hour', 'weekday', 'ADDRTYPE','year','month']
data = pd.get_dummies(df,columns=dummy_columns)
data.head()
```

### 3.3 Machine learning models

We will try several variants of machine learning models.

* Logistic Regression
* Random Forests



#### 3.3.1 Creating Balance

First of all, we need to find a way to deal with the imbalanced-ness of data. This can be handled via weights for logistic regression reflecting the frequency of occurrence.

```python tags=["hide_input"]
weight1 = len(data) / (len(data[data['SEVERITYCODE'] == '1'])*2)
weight2 = len(data) / (len(data[data['SEVERITYCODE'] == '2'])*2)
weight2b = len(data) / (len(data[data['SEVERITYCODE'] == '2b'])*2)
weight3 = len(data) / (len(data[data['SEVERITYCODE'] == '3'])*2)

print("Weight1: " + str(weight1))
print("Weight2: " + str(weight2))
print("Weight2b: " + str(weight2b))
print("Weight3: " + str(weight3))

weights = {'1':weight1, '2':weight2, '2b':weight2b, '3':weight3}
```

#### 3.3.2 Train-test-split

Secondly, we need to split the data into a train and test set.

```python tags=["hide_input"]
X = data.drop('SEVERITYCODE', axis=1).values
#X = preprocessing.StandardScaler().fit(X).transform(X)
y = data['SEVERITYCODE']
```

```python tags=["hide_input"]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
```

#### 3.3.3 Logistic Regression

Here, we model the probability of a feature set to result in either category. This algorithm is designed for binary decision problems, which is the case for this data set and the posed question.

__Model tuning__ We tune model performance by testing several hyperparameters using RandomizedSearchCV.


```python tags=["hide_input"]
LR = LogisticRegression(class_weight=weights)
LR.fit(X_train,y_train)
```

We try a first, naive, logistic regression without tuning.

```python tags=["hide_input"]
y_hat_lg = LR.predict(X_test)
print("Classification Report: Logistic Regression")
print(classification_report(y_test, y_hat_lg))
print("Accuracy: " + str(round(metrics.accuracy_score(y_test, y_hat_lg, normalize=True)*100,2)) + "% (correctly classified test data)")
print("F1: " + str(round(metrics.f1_score(y_test, y_hat_lg, average='weighted'),6))+ " (weighted average of recall and precision)")

```

```python tags=["hide_input"]
reg_params = {
    'penalty': ['l2'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['newton-cg','saga','lbfgs']
}

scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')

randomized_cv = RandomizedSearchCV(
    LogisticRegression(class_weight=weights),
    param_distributions=reg_params,
    cv=3,
    scoring=scorer,
    verbose=5,
    n_jobs=-1,
    random_state=0
)

randomized_cv.fit(X_train,y_train)
```

```python tags=["hide_input"]
y_hat_lg = randomized_cv.predict(X_test)
print("Classification Report: Logistic Regression")
print(classification_report(y_test, y_hat_lg))
print("Accuracy: " + str(round(metrics.accuracy_score(y_test, y_hat_lg, normalize=True)*100,2)) + "% (correctly classified test data)")
print("F1: " + str(round(metrics.f1_score(y_test, y_hat_lg, average='weighted'),6))+ " (weighted average of recall and precision)")

```

Note that in this optimized version, recall for "2b" and "3" is better.


#### 3.3.4 Random Forests

Random forests are a good candidate because they are very versatile and will allow us to work with out categorical data.

```python tags=["hide_input"]
rf_params = {
    'max_depth': [2, 5, 10, 20],
    'max_features': [5, 7, 10, 15],
    'n_estimators': [100, 200, 500, 700],
    'min_samples_split': [10, 15, 20]
}

scorer = metrics.make_scorer(metrics.f1_score, average = 'weighted')

randomized_cv = RandomizedSearchCV(
    RandomForestClassifier(class_weight=weights),
    param_distributions=rf_params,
    cv=3,
    scoring=scorer,
    verbose=5,
    n_jobs=-1,
    random_state=0
)

randomized_cv.fit(X_train,y_train)
```

We can then extract the best model and fit the data.

```python tags=["hide_input"]
RF = randomized_cv.estimator
RF.fit(X_train, y_train)
```

```python tags=["hide_input"]
y_hat_rf = randomized_cv.predict(X_test)
print("Classification Report: Random Forest")
print(classification_report(y_test, y_hat_rf))
print("Accuracy: " + str(round(metrics.accuracy_score(y_test, y_hat_rf, normalize=True)*100,2)) + "% (correctly classified test data)")
print("F1: " + str(round(metrics.f1_score(y_test, y_hat_rf, average='weighted'),6))+ " (weighted average of recall and precision)")

```

<!-- #region -->
So, random forests beat logistic regression in the F1-score. However, the recall for 2b and 3 is too low, hence we still choose Logistic regression.


## 4. Results
Out of our two approaches, logistic classification and random forests, we pick  logistic classification because of the better f1 score performance and the fact that the recall is better for severe injuries and fatalities.

### 4.1  Confusion matrix

The confusion matrix tells us how well our algorithm classifies our test data, e.g. for a given severity code, say fatalities, how many does the algorithm get right and puts into category 3 and how many does it put into other categories?
<!-- #endregion -->

```python tags=["hide_input"]
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat_lg, labels=['1','2', '2b','3'])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['severity code=1','severity code=2','severity code=2b','severity code=3'],normalize= False,  title='Confusion matrix')
```

### 4.2. Feature importance

An interesting side result of the machine learning models we used is that we can extract which features that is which attributes of the accidents play an important role in determining the accident's severity. We get both negative and positive factors: The higher a positive factor, the bigger the higher the likelyhood of an accident of this severity. A negative factor means that this severity is much less likely to happen in the presence of this flag.

```python tags=["hide_input"]
# Severity 1
importances = LR.coef_[0] #[abs(i) for i in LR.coef_[3]]#RF.feature_importances_] 0 corr to 1
data.drop('SEVERITYCODE', axis=1).columns.values
ax = sns.barplot(data.drop('SEVERITYCODE', axis=1).columns.values, importances)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.figure.set_figwidth(20)
ax.set_title("Severity Code 1 - Property damage")
```

__Observations__

Property damage only accidents will __likely happen__ under the following conditions:

* Ice, snow/slush,
* Standing water,
* in alleys,
* between 2-4am.

Property damage only accidents are __less likely to happen__ under the following conditions:

* Blocks and intersections,
* in presence of speeding or intoxicated drivers.


```python tags=["hide_input"]
# Severity 2
importances = LR.coef_[1] #[abs(i) for i in LR.coef_[3]]#RF.feature_importances_] 0 corr to 1
data.drop('SEVERITYCODE', axis=1).columns.values
ax = sns.barplot(data.drop('SEVERITYCODE', axis=1).columns.values, importances)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.figure.set_figwidth(20)
ax.set_title("Severity Code 2 - (Light) injuries")
```

__Observations__

Property damage only accidents will __likely happen__ under the following conditions:

* intersections and blocks,
* oil present,
* in presence of speeding or intoxicated drivers.

Property damage only accidents are __less likely to happen__ under the following conditions:

* ice, snow/slush, standing water
* in alleys,
* dark and no lights.

```python tags=["hide_input"]
# Severity 2b
importances = LR.coef_[2] #[abs(i) for i in LR.coef_[3]]#RF.feature_importances_] 0 corr to 1
data.drop('SEVERITYCODE', axis=1).columns.values
ax = sns.barplot(data.drop('SEVERITYCODE', axis=1).columns.values, importances)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.figure.set_figwidth(20)
ax.set_title("Severity Code 2b - Serious injuries")
```

__Observations__

Property damage only accidents will __likely happen__ under the following conditions:

* intersections and blocks,
* dry or wet roads,
* in presence of speeding or intoxicated drivers.

Property damage only accidents are __less likely to happen__ under the following conditions:

* snow/slush and standing water,
* in alleys,
* daylight, dusk and dawn.

```python tags=["hide_input"]
# Severity 3
importances = LR.coef_[3] #[abs(i) for i in LR.coef_[3]]#RF.feature_importances_] 0 corr to 1
data.drop('SEVERITYCODE', axis=1).columns.values
ax = sns.barplot(data.drop('SEVERITYCODE', axis=1).columns.values, importances)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.figure.set_figwidth(20)
ax.set_title("Severity Code 3 - Fatalities")
```

__Observations__

Property damage only accidents will __likely happen__ under the following conditions:

* intersections and blocks,
* dry, wet or icy roads,
* bad weather conditions,
* in presence of speeding or intoxicated drivers.

Property damage only accidents are __less likely to happen__ under the following conditions:

* snow/slush, dirt or standing water on the road,
* in alleys.

<!-- #region -->

## 5. Discussion


### 5.1 Modeling

Our models work acceptably well to recognize code severe accidents. From a business point of view, correctly classifying higher severity accidents is of higher priority, so one might want to adapt the weights in the classifiers so that they get correctly classified at the expense of potentially misclassifying the lower severities.

Another way of improving our results would be to try and introduce numerical scales for adversity for road, light and weather conditions.

### 5.2 Accident severity

The most striking feature we notice in the feature importance analysis is the importance of intersections. They pose a major risk for injuries in accident.

Both speeding and driving under influence have been identified as high impactful for higher severity. While this is not something drivers themselves can influence for the other drivers, at least this can be given to them as warnings to adapt their own behavior. On the contrary, for traffic regulators, they might wish to instate higher regulatory measurements such as speed traps or mobile units testing drivers for intoxication.

Interestingly, drivers seem to already adapt their behavor already to obviously adverse road, light and weather conditions such as snowing, but fail to take into account medium type conditions such as rain where most severe accidents happen as well as during 'mild' conditions where these conditions seem to not play a big role.

From a timing perspective, the hours during the day are most safe, while driving at night, particularly around 1am seems to be most dangerous.



## 6. Conclusion

In this analysis, we have analyzed the question of what impacts accident severity based on the overall conditions the driver is
facing. We have used as basis the data collected by the Seattle police departement from 2004 to today. We have conducted
a data analysis and selected features that had most impact on the accident severity.

As predictive models, we have used logistic regression and random forests. Random forests performed better by f1 scoring, however recall of the higher categories (correctly identifying serious injuries and fatalities) was not good enough. So we picked logistic regression.

Both models suffer from lower recall for type 2 classifications. In a future analyis,this may be mitigated by using different weights.

Subsequently, we have analyzed feature importance and concluded that speeding, driving intoxicated, intersections and blocks are increasing one's risk in accidents.

<!-- #endregion -->
