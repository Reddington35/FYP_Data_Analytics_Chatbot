from datetime import datetime

import spacy as sp
from spacy.matcher import PhraseMatcher
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt

nlp = sp.load("en_core_web_lg")

# World Dataset operations using different types of charts
# used as a prototype function for basic graphing using the Pandas library
# not used in final Build
def make_line_chart(dataset,entry_field,exit_field,target,region,location,start_date,end_date,color):
     # read in dataframe
     df = pd.read_csv(dataset['Location'],index_col=False)
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     #print("result is ",df)
     # converts date string to date object using the pandas library
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     # Filters dataframe by region, location, start date and end date
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (df['date'] < end_date)]
     # sets plot using target varible, color set in perameters
     plt.plot(df[target],color=color)
     # sets the title of the chart
     plt.title(target, fontsize=14)
     # labels the x values
     plt.xlabel("From "+ start_date + " to " + end_date, fontsize=14)
     # labels the y values
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     # sets to grid view
     plt.grid(True)
     # displays plot
     plt.show()

#make_line_chart("datasets/covid_world.csv","continent","location","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

# Function for plotting a Scatter Plot
# Not used in final Build
def make_scatterplot(dataset,entry_field,exit_field,x,y,region,location,start_date,end_date,color):
     df = pd.read_csv(dataset,index_col=False)
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     print("result is ", df)
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (
                  df['date'] < end_date)]
     plt.scatter(df[x],df[y], color=color)
     plt.title("comparing " + x + " to " + y, fontsize=14)
     plt.xlabel("From " + start_date + " to " + end_date, fontsize=14)
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     plt.grid(True)
     plt.show()

#make_scatterplot("datasets/covid_world.csv","continent","location","total_deaths","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

# Function for plotting Bar Chart
def make_barchart(dataset,entry_field,exit_field,x,y,region,location,start_date,end_date,color):
     # reads in dataframe
     df = pd.read_csv(dataset,index_col=False)
     # seperates df by region or location
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     print("result is ", df)
     # using date time object to fetch date
     df['date'] = pd.to_datetime(df['date'])
     # function to set index of date from df
     df = df.set_index(df['date'])
     # df being partitioned her by ,region,location,start and end date
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (
             df['date'] < end_date)]
     # Bar Graph function
     plt.bar(df[x], df[y], color=color)
     plt.title("comparing " + x + " to " + y, fontsize=14)
     plt.xlabel("From " + start_date + " to " + end_date, fontsize=14)
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     plt.grid(True)
     plt.show()

#make_barchart("datasets/covid_world.csv","continent","location","date","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

# Function for plotting Histogram
# Not used in final build
def make_histogram(dataset,entry_field,exit_field,x,region,location,start_date,end_date,color):
     df = pd.read_csv(dataset, index_col=False)
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     print("result is ", df)
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (
             df['date'] < end_date)]
     # Histogram function
     df.hist(x, figsize=[12,12],bins=12,color=color)
     plt.title("Title:" + x, fontsize=14)
     plt.xlabel("From " + start_date + " to " + end_date, fontsize=14)
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     plt.grid(True)
     plt.show()
#make_histogram("datasets/covid_world.csv","continent","location","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

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

def encoding(X):
    float_columns = list(X.select_dtypes('float64').columns)
    categorical_columns = list(X.select_dtypes('int64').columns)

    pipeline = ColumnTransformer([
        ('num', StandardScaler(), float_columns),
        ('cat', OneHotEncoder(), categorical_columns),
    ])

    encoded_X = pipeline.fit_transform(X)
    return encoded_X
#
# df.head()
# df = encoding(df)
# df.head()no

# timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
# print(timestamp)


