## Data understanding:

<!-- In this phase, you need to collect or extract the dataset from various sources such as csv file or SQL database. Then, you need to determine the attributes (columns) that you will use to train your machine learning model. Also, you will assess the condition of chosen attributes by looking for trends, certain patterns, skewed information, correlations, and so on.
-->

The given data set contains all collisions provided by Seattle police departement and recorded by Traffic Records. The level of aggregation is
weekly. The timeframe is 2004 to today.

(See metadata from the SPD Collision data set.)

We need to predict severity of the accident. The following codes are given:

3—fatality
2b—serious injury
2—injury
1—prop damage
0—unknown 

(1) Therefore, we can drop already the 'O'-valued columns, since no information is contained in there.

(2) We see that the data is highly unbalanced, e.g. 133 776  of the 200.082 remaining rows are 'mild', e.g. only property damage.
In order to get meaningful information on the severity of the severe (that is more than property damage), I will do a two step analyis

1. Determine if an accident is mild (property damage only) or severe (injuries or deaths)
2. If severe, determine the severity






'SEVERITYCODE',
'SEVERITYCODE.1', 
'SEVERITYDESC', 



'X',
'Y',
'OBJECTID',
'INCKEY', 
'COLDETKEY', 
'REPORTNO',
'STATUS', 
'ADDRTYPE', 
'INTKEY', 
'LOCATION', 
'EXCEPTRSNCODE',
 EXCEPTRSNDESC', 

'COLLISIONTYPE',
'PERSONCOUNT', 
'PEDCOUNT', 
'PEDCYLCOUNT', 
'VEHCOUNT', 
'INCDATE',
'INCDTTM', 
'JUNCTIONTYPE', 
'SDOT_COLCODE', 
'SDOT_COLDESC',
'INATTENTIONIND', 
'UNDERINFL', 
'WEATHER', 
'ROADCOND', 
'LIGHTCOND',
'PEDROWNOTGRNT', 
'SDOTCOLNUM', 
'SPEEDING', 
'ST_COLCODE', 
'ST_COLDESC',
'SEGLANEKEY', 
'CROSSWALKKEY', 
'HITPARKEDCAR'

