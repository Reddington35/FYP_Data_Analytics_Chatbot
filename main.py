import pandas as pd
import numpy as np
import spacy as sp
import sklearn
from sklearn.ensemble import RandomForestClassifier

sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")


from pickle import dump,load
data = pd.read_csv("covid.csv",sep=',')
data = data[["SingleDoseCum","SingleDose","Dose1Cum","Dose2Cum","Dose2",
             "Dose1","FullyVacc","PartialPercent","FullyPercent"]]

predict = "SingleDose"

x = np.array(data.drop([predict],axis=1))
y = np.array(data[predict])

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.10,random_state=1)
model = RandomForestClassifier(min_samples_split=3,n_estimators=6,max_depth=4,max_features=6)

model.fit(x_train, y_train)
#dump(model,open('model.csv','wb'))
accuracy = model.score(x_test, y_test)

predictions = model.predict(x_test)
for x in range(len(predictions)):
    print("Predictions: " + str(predictions[x]) + "\n",
          "features test Data: " + str(x_test[x]) +"\n",
          "Target test Data: " + str(y_test[x]) + "\n")

print(accuracy)









