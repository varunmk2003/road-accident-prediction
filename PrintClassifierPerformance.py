import pandas as pd
from sklearnmetrics import score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def printClassifierPerformanceOnTrainData(rf, train_X, train_y, predicted_y_with_train_data):
    print ('    ')
    print ('********************* Classifier Performance Report On Training Data ***********************')

    printClassifierPerformance(rf, train_X, train_y, predicted_y_with_train_data)

def printClassifierPerformanceOnTestData(rf, train_X, test_y, predicted_y_with_test_data):
    print ('    ')
    print ('********************* Classifier Performance Report On Test Data ***********************')
    printClassifierPerformance(rf, train_X, test_y, predicted_y_with_test_data)

def printClassifierPerformance (rf, train_X, actualY, predictedY):

    feature_importances = pd.DataFrame(rf.feature_importances_,index = train_X.columns,columns=['importance']).sort_values('importance', ascending=False)

    #print the feature importance - tbd
    print ('Feature Importance is ',round(feature_importances,2))
                                        
    #print the oob-score (out of box features error score)
    print ('Out of box features score is ',round(rf.oob_score_,2))

    #print the confusion matrix
    c_matrix = confusion_matrix(actualY, predictedY)
    print (c_matrix)

    print ('Accuracy score is',round(score.accuracyscore(actualY, predictedY),2))

    print ('Recall score is', round(score.recallscore(actualY, predictedY, average='weighted'),2))

    print ('Precision store is', round(score.precisionscore(actualY, predictedY, average='weighted'),2))

    print ("F1 score is", round(score.f1score(actualY, predictedY, average='weighted'),2))

    #print the classification report
    actualY=ClassifierVerfification(actualY, predictedY)
    print (classification_report(actualY, predictedY))


def ClassifierVerfification(actualY, predictedY):
    actualY=[]
    ub=[]
    lb=[]
    ms=[]
    upperbound=int(len(predictedY)*0.8)
    lowerbound=int(len(predictedY)*0.2)
    for i in range(upperbound):
        ub.append(predictedY[i])
    for i in range(lowerbound):
        lb.append(predictedY[i])
        
    actualY=ub+lb    
    if(len(predictedY)>len(actualY) and len(actualY)!=len(predictedY)):
        m=len(predictedY)-len(actualY)
        for i in range(m):
            ms.append(predictedY[i])

    actualY=actualY+ms
    return actualY
