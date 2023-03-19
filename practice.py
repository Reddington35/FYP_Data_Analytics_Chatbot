from datetime import datetime

import spacy as sp
from spacy.matcher import PhraseMatcher
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

nlp = sp.load("en_core_web_lg")

# matcher = PhraseMatcher(nlp.vocab)
# matcher.add("Ireland", [nlp("ireland"),nlp("irish")])
# matcher.add("EU", [nlp("arack Obama"),nlp("Baracko")])
# matcher.add("World", [nlp("Scatter plot"), nlp("scatterplot"),nlp("scat")])
# matcher.add("Who",[nlp("Scatter plot"), nlp("scatterplot"),nlp("scat")])
# doc = nlp("Ireland is in ireland which is irish")
# matches = matcher(doc)
# matchesS = [nlp.vocab.strings[s[0]] for s in matches]
# #print(nlp.vocab.strings)
# #print(matchesS)
# target = "Ireland"
# if target in matchesS:
#     print(target," Found!")
#print(matches)

# #initilize the matcher with a shared vocab
# matched = PhraseMatcher(nlp.vocab)
# #create the list of words to match
# plot_types = ['scatter plot','bar chart','histogram',"bar"]
# #obtain doc object for each word in the list and store it in a list
# patterns = [nlp(plots) for plots in plot_types]
# #add the pattern to the matcher
# matched.add("plot_patern", patterns)
# #process some text
# doc = nlp("in this document there seems to be a scatter plot and a bar chart")
# matching = matched(doc)
# print(matching)
# for match_id, start, end in matching:
#  span = doc[start:end]
#  print(span.text)



# doc = nlp("color pink black blue")
# # for token in doc:
# #     print(token.dep_)


# ML Methods
def RandomForest(target, labels, dataset):
    print(dataset)
    newLabels = labels
    newLabels.append(target)
    print("Labels: " + str(newLabels))
    newDataset = dataset[newLabels].copy()
    print("Dataset: ")
    print(newDataset)
    cleanDataset = newDataset.dropna()
    # cleanDataset = cleanDataset.head(50000)
    print(cleanDataset)

    X = cleanDataset[labels]
    y = cleanDataset[target]
    print(X)
    print(y)
    # use min_max scaler to normalise and scale the data
    scaler = MinMaxScaler()
    print(scaler.fit(X,y))
    MinMaxScaler()
    print(scaler.data_max_)
    print(scaler.transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators=10, max_depth=6, criterion="entropy")
    print("Fitting dataset")
    clf.fit(X_train, y_train)
    print("predicting dataset")
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#RandomForest("Cases - cumulative total","Deaths - newly reported in last 24 hours","datasets/WHO_covid.csv")

def encoding(dataset):
    df = pd.read_csv(dataset, index_col=False)
    print(df['weekly_hosp_admissions'])
    feat = df.head(0).applymap(str)
    new_features = feat.columns.values
    world = pd.DataFrame(new_features,columns=['new_features'])

    print(df['location'].dtype)
    label_encoder = LabelEncoder()
    world['new_features_Cat'] = label_encoder.fit_transform(world['new_features'])
    print(world['new_features_Cat'])

encoding("datasets/covid_World.csv")


# timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
# print(timestamp)


