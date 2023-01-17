import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm

data = pd.read_csv("datasets/covid_EU.csv", sep=',')
data = data[["YearWeekISO","ReportingCountry","Denominator","NumberDosesReceived","NumberDosesExported","FirstDose",
             "FirstDoseRefused","SecondDose","DoseAdditional1","DoseAdditional2","DoseAdditional3","UnknownDose","Region",
             "TargetGroup","Vaccine","Population"]]

predict = "Denominator"

x = np.array(data.drop([predict],axis=1))
y = np.array(data[predict])

#x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.10,random_state=1)
#model = RandomForestClassifier(min_samples_split=3,n_estimators=6,max_depth=20,max_features=6)

parameters = {"probability" : True}
svm_classifier = svm.SVC()
gridsearch = GridSearchCV(svm_classifier,parameters)
gridsearch.fit(x,y)
print(sorted(gridsearch.cv_results_.keys()))
print(gridsearch.best_params_)
print("Accuracy:"+ str(gridsearch.best_score_))

#model.fit(x_train, y_train)
#dump(model,open('model.csv','wb'))
#accuracy = model.score(x_test, y_test)

#predictions = model.predict(x_test)
#for x in range(len(predictions)):
 #   print("Predictions: " + str(predictions[x]) + "\n",
  #        "features test Data: " + str(x_test[x]) +"\n",
   #       "Target test Data: " + str(y_test[x]) + "\n")

#print(accuracy)import pandas as pd
