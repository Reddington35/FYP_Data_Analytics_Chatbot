import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("datasets/covid_Ireland.csv", sep=',')
data = data[["X","Y","VaccinationDate",
             "VaccineText","Dose1","Dose2","SingleDose","Dose1Cum","Dose2Cum","SingleDoseCum","PartiallyVacc","FullyVacc","PartialPercent",
             "FullyPercent","ObjectId"]]

predict = "PartialPercent"

x = np.array(data.drop([predict],axis=1))
y = np.array(data[predict])

#x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.10,random_state=1)
#model = RandomForestClassifier(min_samples_split=3,n_estimators=6,max_depth=20,max_features=6)

parameters = {"min_samples_split" :[2,10],"max_depth" : [2,10],"criterion" : ["gini"]}
gridsearch = GridSearchCV(RandomForestClassifier(),parameters)
gridsearch.fit(x,y)
print(sorted(gridsearch.cv_results_.keys()))
print(gridsearch.best_params_)

#model.fit(x_train, y_train)
#dump(model,open('model.csv','wb'))
#accuracy = model.score(x_test, y_test)

#predictions = model.predict(x_test)
#for x in range(len(predictions)):
 #   print("Predictions: " + str(predictions[x]) + "\n",
  #        "features test Data: " + str(x_test[x]) +"\n",
   #       "Target test Data: " + str(y_test[x]) + "\n")

#print(accuracy)