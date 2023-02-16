import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import NLP

# The data_summary method provides the user with a clean representation of the data being provided by the Datasets
# they may be interested in analysing, this method provides the user with the number of Rows,columns,and shape of the
# desired Dataset they are working with while also providing the type of Data provided in each label
def data_summary(dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    print("Data Summary:")
    print("The number of rows in this Dataset are " + str(len(df.index)))
    print("The number of columns in this dataset are " + str(len(df.columns)))
    print("The shape of this Dataset is " + str(df.shape))
    print("\nThe available Data types contained in this Dataset: \n")

    for i in df.head(0):
        print("Label: " + i + " " + "|Type: " + str(df.dtypes[i]) + " |value contained in first Row:" + str(df[i][0]) + "|")
    print("\n*****************************************************************************************************************\n")
    return df

def data_plot(dataset, username, request):
    df = pd.read_csv(dataset, sep=',')

def data_ml(dataset, username, request):
    df = pd.read_csv(dataset['Location'], sep=',')
    labels = printChoices(df)

    print("Colin: Give target feature please " + username)
    choice = input(username)
    print(choice)
    # make sure we get a non-empty list from label match

    target = NLP.label_match(choice, labels)[0]
    labels = printChoices(df)
    print("Colin: Give parameters please " + username)
    choice = input(username)
    params = NLP.label_match(choice, labels)

    labels = []

    for label in params:
        labels.append(label)

    globals()[request['def']](target, labels, df)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators=10, max_depth=6, criterion="entropy")
    print("Fitting dataset")
    clf.fit(X_train, y_train)
    print("predicting dataset")
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
def NaiveBayes(target, labels, dataset):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = GaussianNB()
    print("Fitting dataset")
    clf.fit(X_train, y_train)
    print("predicting dataset")
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def SupportVectorMachine(target, labels, dataset):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = SVR(kernel='linear')
    print("Fitting dataset")
    clf.fit(X_train, y_train)
    print("predicting dataset")
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Utility
def printChoices(df):
    labels = df.columns.values
    labelView = ""
    # for loop uses modulus to print the label and move to the next line of the print
    # with every 6 entries of the datasets labels
    for i in range(0, len(labels)):
        if i % 7 == 6:
            labelView += " |\n"
        labelView += " | " + labels[i]
    print("Available features are: \n")
    labelView += " |\n"
    print(labelView)
    return labels

# label_selection method is used to search the requested Dataset and print all available labels represented in the
# dataset
def feature_selection(dataset):
    #opens the dataframe using pandas library
    df = pd.read_csv(dataset['Location'], sep=',')
    labels = df.columns.values
    labelView = ""
    # for loop uses modulus to print the label and move to the next line of the print
    # with every 6 entries of the datasets labels
    for i in range(0, len(labels)):
        if i % 7 == 6:
            labelView += " |\n"
        labelView += " | " + labels[i]
    print("Available Features: \n")
    labelView += " |\n"
    print(labelView)

def load_dataset(statement,queryTexts,username):
    # Getting dataset section
    content = False
    while not content:
        # locates the dataset if user input matches the available dataset
        dataset = NLP.phrase_match(statement, queryTexts["Datasets"])

        # Get target classification section
        print("Selection")
        print("-------------")
        print("Dataset: " + dataset['Title'])
        print("Location: " + dataset['Location'])

        # print("Colin: Are you happy with these details y/n")
        # checks to see if the user is happy with their selection
        # if yes chatbot will move to the next query otherwise it will ask the user
        # for further clarification
        contentAnswer = decision_handler("Colin: Are you happy with these details y/n",username)
        if contentAnswer:
            content = True
        else:
            print("Colin: Please rephrase and I will do my best to understand your choice")
            statement = input(username)
            content = False
        return dataset

def change_Datasets(dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    return df

# decision_handler method is used to manage basic yes or no patterns in questions
# no nlp is used here just takes in a question and user input as parameters and manages the response the user provides
# if no yes/no response is given the method will ask the user to simply respond yes/no
def decision_handler(question,username):
    print(question)
    summary = input(username)
    while summary != "yes" and summary != 'y' \
            and summary != "no" and summary != 'n':
        print("Colin: could you please enter yes or no as your response")
        print(question.lower())
        summary = input(username)
    if summary == "yes" or summary == 'y':
        return True
    else:
        return False

# method for identifying user login details
def user_login(login):
    login = login + " : "
    return login

# find_target method searches the head of the dataset (the labels) and checks to see if their a match
# with the user input
def find_target(user_input, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    if user_input in df.head(0):
        print("found")
        target = user_input
        return target
    else:
        print("not found")

