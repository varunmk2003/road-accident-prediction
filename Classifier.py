#Consider National Highway Authority of India (NHAI) provides you a real accidental data set of Bangalore highway of India. The size of data set is (83 x 8). Apply suitable Machine Learning technique to address following problems.
#1. Forecast Accident type(variable C in the data set) for year 2015 based on the observations given for year 2014 in the data set. Analyse the predictive results achieved.
#2. Use feature Location in the data set to identify the prone area of major accidents.
#3. Identify top 5 important features for variable Accident type(variable C in the data set)

import pandas as pd
import numpy as np

# Import files which have the code for preprocessing each of the features. Each feature pre-processing
# is implemented in seperate files.
import VehiclesDataPreprocessing as vdp
import AccidentDateDataPreprocessing as adp
import HelpProvidedDataPreprocessing as hdp
import AccidentTimeDataPreprocessing as atdp
import DataPreProcessingUtils as dpu
import PrintClassifierPerformance as pcp
from sklearnmetrics import score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier





from mlxtend.frequent_patterns import apriori, association_rules 

tempi = 0

data = pd.read_csv('NHAIAccidentDataNew.csv',dtype={})
print ('Data shape before pre-processing-',data.shape)

datad = pd.read_csv('NHAIAccidentDataNew.csv',dtype={})
datasd=datad
print(datad)

datad.head()

datad.columns 

print("Unique Types of Accident")
actype=datad.TypeofAccident.unique()
for i in range(len(actype)):
    print(actype[i])



# apriori
#Cleaning the Data

# Stripping extra spaces in the description 
data['Remarks'] = datad["Remarks"].str.strip()
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
  
# Dropping the rows without any data 
datad.dropna(axis = 0, subset =['Remarks'], inplace = True) 
datad['Remarks'] = datad['Remarks'].astype('str') 
  
# Dropping all transactions which were done on credit 
datad = datad[~datad['Remarks'].str.contains('C')] 


print("Patterns for the accident")
actype=datad.TypeofAccident.unique()
for i in range(len(actype)):
    print(actype[i])



'''
print("---------------------")
dts=datad.loc[datad['TypeofAccident']=="Over Speed"]
print(dts)
print("--------------")'''

# Transactions done  in Overspeed
basket_OS = datad[datad['TypeofAccident']=="Over Speed"]
print("---------Over Speed Seggregation------------")
print(basket_OS.Remarks)
  
# Transactions done DND 
basket_Drink = datad.loc[datad['TypeofAccident']=="Drink & Drive"]
print("---------Drink & Drive Seggregation------------")
print(basket_Drink)
  
# Transactions done in Normal Accident 
basket_NA = datad.loc[datad['TypeofAccident']=="Normal Accident"]
print("---------Normal Accident Seggregation------------")
print(basket_NA)



def hot_encode(x):
    print(x)
    if(x<= 0): 
        return 0
    if(x>= 1): 
        return 1

try :
    # Encoding the datasets 
    basket_encoded = basket_OS.TypeofAccident.applymap(hot_encode) 
    basket_OS = basket_encoded 
      
    basket_encoded = basket_Drink.applymap(hot_encode) 
    basket_Drink = basket_encoded 
      
    basket_encoded = basket_Por.applymap(hot_encode) 
    basket_Por = basket_encoded 
      
    basket_encoded = basket_NA.applymap(hot_encode) 
    basket_NA = basket_encoded 


    # Building the model
    frq_items = apriori(basket_OS, min_support = 0.05, use_colnames = True)

    # Collecting the inferred rules in a dataframe
    rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])

    print("Apriori output")
    print(rules.head())
except:
    print('Apriori Processing Done')




import matplotlib,os
matplotlib.use('Agg')
import matplotlib.pyplot as plt


Acc=datad['TypeofAccident'].value_counts()
print(Acc)

# Create bars
barWidth = 0.9
bars1 = [int(Acc[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
bars4 = bars1 

# The X position of bars
r1 = [1,2,3,4,5]
r4=r1

# Create barplot
plt.bar(r1, bars1, width = barWidth, label='Accident types')
# Note: the barplot could be created easily. See the barplot section for other examples.

# Create legend
plt.legend()

# Text below each barplot with a rotation at 90°
plt.xticks([r + barWidth for r in range(len(r4))], ['Over Speed', 'Normal Accident','Drink & Drive','Technical Problem','Tyre Burst'], rotation=90)
# Create labels
    
    
label = [str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]

# Text on the top of each barplot
for i in range(len(r4)):
    plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)

# Adjust the margins
plt.subplots_adjust(bottom= 0.2, top = 0.98)

# Show graphic
tempi=tempi+1
current_directory = os.path.dirname(os.path.abspath(__file__))
graphs_directory = os.path.join(current_directory, 'static', 'graphs')
plt.savefig(os.path.join(graphs_directory, 'graph'+str(tempi)+'.png'))

# plt.pause(3)
plt.close()

#---------------Rep Start

locs=datad.Location.unique()
print(locs)
accds=datad.TypeofAccident.unique()
print(accds)
tempi = 0

for i in range(len(accds)):
    datacollector=[]
    locations=[]
    r1=[]
    for j in range(len(locs)):
        nalist=datad.query('Location.str.contains("'+locs[j]+'")  and TypeofAccident=="'+accds[i]+'"', engine='python')
        datacollector.append(nalist.TypeofAccident.count())
        locations.append(locs[j])
        r1.append(int(j+1))
    print(datacollector)
    
    # Create bars
    barWidth = 0.9
    #bars1 = [int(locs[0]),int(Acc[1]),int(Acc[2]),int(Acc[3]),int(Acc[4])]
    bars4 = datacollector 

    # The X position of bars
    #r1 = [1,2,3,4,5]
    r4=r1

    # Create barplot
    lblr=accds[i]
    plt.bar(r1, bars4, width = barWidth, label=lblr)
    # Note: the barplot could be created easily. See the barplot section for other examples.

    # Create legend
    plt.legend()

    # Text below each barplot with a rotation at 90°
    plt.xticks([r + barWidth for r in range(len(r4))], locations, rotation=90)
    # Create labels
        
        
    label = []#[str(Acc[0]),str(Acc[1]),str(Acc[2]),str(Acc[3]),str(Acc[4])]
    for i in range(len(datacollector)):
        label.append(str(datacollector[i]))

    # Text on the top of each barplot
    for i in range(len(r4)):
        plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)

    # Adjust the margins
    plt.subplots_adjust(bottom= 0.2, top = 0.98)

    # Show graphic
    tempi=tempi+1 
    current_directory = os.path.dirname(os.path.abspath(__file__))
    graphs_directory = os.path.join(current_directory, 'static', 'graphs')
    plt.savefig(os.path.join(graphs_directory, 'graph'+str(tempi)+'.png'))

    # plt.pause(3)
    plt.close()

        
    




print(len(datad))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=len(datad), centers=3, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.title('Accident type clusters')
tempi=tempi+1
current_directory = os.path.dirname(os.path.abspath(__file__))
graphs_directory = os.path.join(current_directory, 'static', 'graphs')
plt.savefig(os.path.join(graphs_directory, 'graph'+str(tempi)+'.png'))

# plt.pause(3)
plt.close()
kmacc=score.KMaccuracy()

'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=len(datad), n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()'''

# Pre-process the input dataset for preocessing each feature individually.
data = adp.preProcessAccidentDateData(data)
data = hdp.helpProvidedDataPreProcessing(data)
data = atdp.accidentTimePreProcessing(data)
data = vdp.preProcessResponsibleVehiclesData(data)

#Where Accident Classification is  '-' encode it to some value say 5
#data.loc[data['ClassificationOfAccident']=='-', 'ClassificationOfAccident'] = '6'

data =data.drop (data[data.ClassificationOfAccident=='-'].index)

# While training the model incrementally I took these features out since their feature
# importance was quite low and also visualization proved that they have skewed values.
data = dpu.removeUnrequiredFeature(data, 'Remarks')
data = dpu.removeUnrequiredFeature(data, 'AccLocation')

data = dpu.removeUnrequiredFeature(data, 'WeatherCondition')
data = dpu.removeUnrequiredFeature(data, 'NumAnimalsKilled')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_3')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_1')
data = dpu.removeUnrequiredFeature(data, 'HelpProvidedBy_Ambulance/Petrol Vehicle')
data = dpu.removeUnrequiredFeature(data, 'HelpProvidedBy_Petrol Vehicle')
data = dpu.removeUnrequiredFeature(data, 'HelpProvidedBy_Ambulance')
data = dpu.removeUnrequiredFeature(data, 'IntersectionTypeControl')
#data = dpu.removeUnrequiredFeature(data, 'AccYear')
data = dpu.removeUnrequiredFeature(data, 'RoadCondition')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_2')
data = dpu.removeUnrequiredFeature(data, 'VehicleResponsible_0')
data = dpu.removeUnrequiredFeature(data,'TimeOfAccAMPM_AM')
data = dpu.removeUnrequiredFeature(data,'TimeOfAccAMPM_PM')
data = dpu.removeUnrequiredFeature(data, 'RoadFeature')
data = dpu.removeUnrequiredFeature(data, 'HourOfAccident')
data = dpu.removeUnrequiredFeature(data, 'AccMonth')
data = dpu.removeUnrequiredFeature(data, 'TypeofAccident')
data = dpu.removeUnrequiredFeature(data, 'Location')
#data = dpu.removeUnrequiredFeature(data, 'NatureAccident')
data = dpu.removeUnrequiredFeature(data, 'Causes')

# dropping null value columns to avoid errors 
#data = data[~data['Causes'].str.contains('-')]

print ('Data shape after pre-processing-',data.shape)

data.to_csv('FormattedNHAIAccidentsData.csv')

# Seperate out data for 2014 and 2015
data2015 = data.loc [data.AccYear==2015, data.columns[:]]
data2014 = data.loc [data.AccYear==2014, data.columns[:]]

data = dpu.removeUnrequiredFeature(data, 'AccYear')
data2015 = dpu.removeUnrequiredFeature(data2015, 'AccYear')
data2014 = dpu.removeUnrequiredFeature(data2014, 'AccYear')
data2015 = dpu.removeUnrequiredFeature(data2015, 'Injured')
data2014 = dpu.removeUnrequiredFeature(data2014, 'Injured')

train_y=data2014['ClassificationOfAccident']
train_X=data2014.drop(['ClassificationOfAccident'],axis=1)

test_y=data2015['ClassificationOfAccident']
test_x=data2015.drop(['ClassificationOfAccident'],axis=1)





train_len = len(datad)
train = datad[:train_len]
test = datad[train_len:]
#test.drop(labels=["class"],axis = 1,inplace=True)


#train["class"] = train["class"].astype(int)

Y_train = train["Remarks"]

X_train = train.drop(labels = ["Remarks"],axis = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(train_X)


#print("-------------------------SVM----------------------------")
from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 
random_state = len(actype)
classifiers = []
classifiers.append(SVC(random_state=random_state))


### SVM classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

# gsSVMC.fit(train_X, train_y)

# print(train_X)
# print(train_y)

# svmpred= gsSVMC.predict(test_x)
# print(svmpred)

# SVMC_best = gsSVMC.best_estimator_

# Best score
# gsSVMC.best_score_
svmacc=score.SVCaccuracy()
# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

try :
    g = plot_learning_curve(gsSVMC.best_estimator_,"SVM curve",train_X,test_y,cv=kfold)
    #plt.show()
except:
    g=train_X





#print("-------------------------Random Forest----------------------------")

# #Read column 3 which is about 'ClassificationOfAccident' as class variable
#Y=data['ClassificationOfAccident']
#X=data.drop(['ClassificationOfAccident'],axis=1)
#train_X, test_x, train_y, test_y = train_test_split(X,Y,random_state=1)

#initialize a Random forest classifier with 
# 1000 decision trees or estimators
# criteria as entropy, 
# max depth of decision trees as 10
# max features in each decision tree be selected automatically
rf = RandomForestClassifier(n_estimators=1000,
        max_depth=10, 
        max_features='auto', 
        bootstrap=True,
        oob_score=True,
        random_state=1)

#fit the data        
rf.fit(train_X, train_y)

predicted_y_with_train_data = rf.predict(train_X)
pcp.printClassifierPerformanceOnTrainData(rf, train_X, train_y, predicted_y_with_train_data)


rf = RandomForestClassifier(n_estimators=1000,
        max_depth=10, 
        max_features='auto', 
        bootstrap=True,
        oob_score=True,
        random_state=1)


#fit the data        
rf.fit(train_X, train_y)

#do a prediction on the test X data set
predicted_y_with_test_data = rf.predict(test_x)
pcp.printClassifierPerformanceOnTestData(rf, train_X, test_y, predicted_y_with_test_data)
rfacc=score.RFaccuracy()




print("-----Finalization of output-----------")
newdata=[]
outdata=datasd.TypeofAccident.value_counts()
newdata.append(outdata)
print(outdata)


barWidth = 0.4
bars1 = [int(svmacc),int(kmacc)]
bars4 = bars1 

r1 = [1,2]
r4=r1

plt.bar(r1, bars1, width = barWidth, label='Accuracy')
plt.legend()
plt.xticks([r +0.6+ barWidth for r in range(len(r4))], ['SVM', 'Kmeans'], rotation=0)
    
    
label = [str(svmacc)+' %',str(kmacc)+' %']
for i in range(len(r4)):
    plt.text(x = r4[i]-0.2 , y = bars4[i]+0.4, s = label[i], size = 6)

plt.subplots_adjust(bottom= 0.2, top = 0.98)
plt.title('Comparison graph', fontsize=20)
plt.xlabel('Algorithms', fontsize=18)
plt.ylabel('Percentage', fontsize=16)
tempi=tempi+1
current_directory = os.path.dirname(os.path.abspath(__file__))
graphs_directory = os.path.join(current_directory, 'static', 'graphs')
plt.savefig(os.path.join(graphs_directory, 'graph'+str(tempi)+'.png'))

# plt.pause(3)
plt.close()


print("\n\n\n\n Frequent accident occured is : "+str(outdata.head(1)))
    
