# Car accident severity predictor

## 1. Introduction

The objective of this project is to create a way to predict severity of car accidents happening based on road and weather conditions.
That is, our algorithm needs to take as input the conditions that may impact severity of accidents:

* road conditions
* weather conditions
* lighting conditions

and predict the risk profile of potential accidents, i.e. the severity label. A user should then be able to specify the current conditions
of their itinerary and should then get back the severity label of accidents happening.

Example: A wet road should potential lead to higher severity accidents than a dry road.

The main stakeholders are:

* __traffic regulators__: They can make policies (e.g. speed limits, higher controls etc) in order to mitigate risks.
* __drivers__: They can recognize risks in advance and adapt their behavior in order to lower their own personal risks, e.g. drive more carefully or even avoid travelling at at risk conditions all together.

## 2. Data understanding 

### Data Source

The given data set contains all collisions provided by Seattle police departement and recorded by Traffic Records. The level of aggregation is
weekly. The timeframe is 2004 to today.
(See metadata from the SPD Collision data set.)

## Data cleaning

__Feature unknown values__
we need to take out the value=Unknown for the features.


Thirdly, we need to convert the features into numerical values in order for the algorithms to work.

### Label analysis
We need to predict severity of the accident given by the data colum 'SEVERITYCODE'. The following codes are given:

* 3—fatality 
* 2b—serious injury
* __2—injury__
* __1—property damage__
* 0—unknown

Only labels of 1,2 are present in the dataset.



### Feature selection

In order to predict the label, our severity code, we will use the following colums as features.

* 'WEATHER' containing data on the weather such at if it was dry or wet etc.
* 'ROADCOND' containing information whether the road was dry, wet etc.
* 'LIGHTCOND' containing data on lighting, e.g. if it was dark.



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


## 3. Methodology
<!--
Methodology section which represents the main component of the report where you discuss and describe any exploratory data analysis that you did, any inferential statistical testing that you performed, if any, and what machine learnings were used and why.-->

### Exploratory data analysis

todo: pictures of correlations

#### Weather

The following picture  shows the number of occurencies of weather conditions and severity codes in the data.
![Weather conditions](weather1.png)
Since the diagram is dominated by 'Other', 'Overcast', 'Raining' and 'Unknown', here is a version without these categories.
![Weather conditions](weather2.png)

We clearly see, that clear weather, with a large amount of data, favors less severe accidents of type 1 (ratio 2 to 1), while during overcast weather or raining, severe accidents are much more likely.

Interestingly, snowing favors less severe accidents, which might be due to the fact that
people already drive more carefully when it is snowing having already reckognized this
risk while underestimating the impact of rain.

### Lighting


The following picture  shows the number of occurencies of weather conditions and severity codes in the data.
![Lighting conditions](lightcond1.png)
Since the diagram is dominated by 'Other', 'Overcast', 'Raining' and 'Unknown', here is a version without these categories.
![Light conditions](lightcond2.png)


### Road conditions


The following picture  shows the number of occurencies of weather conditions and severity codes in the data.
![Lighting conditions](roadcond1.png)
Since the diagram is dominated by 'Other', 'Overcast', 'Raining' and 'Unknown', here is a version without these categories.
![Light conditions](roadcond2.png)



### Machine learning models

The following models will be employed:

__k-nearest neighbors (KNN)__ 

This algorithm will assign a label to an unknown feature set based on the labels of its k closest neighbors, where closeness is defined as closest to the features.
caveat: does not work well with categorical features

__decision trees__ 

This algorithm will find criteria on the features organized in a tree like way in order to predict the right label category.

__logistic regression__ 

Here, we model the probability of a feature set to result in either category. This algorithm is designed for binary decision problems, which is the case for this data set and the posed question.

### Imbalance of the data

The data is highly imbalanced in favor of severity code 1 (nearly two times as likely):

| SEVERITYCODE | Occurences |
| -- | -- |
| 1 | 136485 |
| 2 | 58188  | 

This will pose an issue in applying or models. A naive model that would just predict 1 everywhere and be right in 2 cases out of 3.
However missing the one injury case (label 2 not recognized) is a very costly error to make.


## 4. Results 

### KNN
??

### Decision tree
??

## Logistic regression
??

### Comparison
Table of accuracy

| KPI | KNN | Decision tree | Logistic regression | 
| --  | --  | -- | -- | 
| Accuracy  | ??  | ?? | ?? |
| Log loss  | ??  | ?? | ?? |
| ??  | ??  | ?? | ?? |

Thus, showing that ?? is the best algorithm for tackling this challenge.



## 5. Discussion 

section where you discuss any observations you noted and any recommendations you can make based on the results.


It might be benefitial in a a further analysis to find numerical values for the severity, such as a severity score based on e.g. the amount of property damage, but also the number and severity of injuries or deathsm as well as numerical values for the features measuring the 'badness' of e.g. the weather (being 'overcast' should be very close to 'clear' while 'raining' should be further away).

Drivers should therefore avoid ??.

NB: it would also be interesting to study the frequency of car accidents in addition to their severity.




## 6. Conclusion

In this study, I analyzed the impact of weather, road and lighting conditions on severity of car accidents. I transformed the
data in such a way such that machine learning models were able to predict the severity (e.g. property damage only or also injuries) based
on a given set of weather, road and lighting conditions.

The best machine learning model turned out to be ??.

Some dangers seem to be well recognized by drivers (snow), while others such as 
rain not as much. 

