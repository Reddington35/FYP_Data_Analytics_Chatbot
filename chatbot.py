import spacy as sp
import json
import pandas as pd
import Data_summary
from category_encoders import one_hot
from sample_chart import make_line_chart
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

# Nlp model chosen for the chatbot
sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

# Open json file
queryTexts = json.load(open('interpretation.json'))

# member variables used in conjunction with json
dataset = -1
target_classification = -1
variables = -1
approach = -1

# This is the main function of the chatbot and will handle any conversational functionality
# this method is primarily used to decipher the Dataset chosen,Approach such as training or plotting
# and location of the requested Dataset from the json file named interpretation.json
# and operate the chatbot with the functions written to improve its mechanics
def chatbot():
    print("Colin: Hi my name is Colin, your Covid-19 chatbot, who am i talking to?")
    login = input("User :")
    username = user_login(login)
    print("Colin: Hello " + username.replace(':',',') + "Which dataset will we be working with today?")
    # statement takes in the user input in lowercase
    statement = input(username)
    statement = statement.lower()

    # Getting dataset section
    content = False

    while not content:
        # locates the dataset if user input matches the available dataset
        dataset = scanJson("Datasets", statement)

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
            print("Colin: Please rephrase and I will do my best to understand it")
            statement = input(username)
            content = False
    not_exit = True

    # Basic Instructions on the services available by the chatbot
    print("Colin: I am a covid-19 chatbot, however I can only provide "
              "help on topics such as:\n - Data summary of the Dataset\n - Graph the Dataset\n - Build a ml model of the"
              " dataset using (Random Forrest, Naive Bayes or SVM)\n - Display the available Features")

    # While loop that checks to see if the user wants to quit the application
    # or choose from four distinct services available
    while not_exit:
        print("Colin: which task would you like me to perform?")
        task = input(username)
        not_exit = check_command(task,username)
        if "summary" in task.lower():
            task_summary = decision_handler("Colin: are you sure you would like to see a summary of "+ dataset["Title"]
                                            + "?",username)
            if task_summary:
                Data_summary.data_summary(dataset['Location'])
        elif "features" in task.lower():
            task_features = decision_handler("Colin: are you sure you would like to view the available features contained in "
                             + dataset["Title"] + " " + username.replace(':',''),username)
            if task_features:
                feature_selection(dataset)
        elif "random forrest" in task.lower():
            random_forrest = decision_handler("Colin: are you sure you would like to perform Random Forrest on " + dataset['Title'] + " "
                             + username.replace(':',''),username)
            if random_forrest:
                train_model("Deaths - cumulative total per 100000 population",
                            ["Cases - cumulative total per 100000 population"], dfDs)
        elif not_exit:
            print("Colin: How can I help you, please be specific with your queries " + username.replace(':', ''))
        elif not_exit is False:
            print("Colin: Goodbye " + username.replace(':',''))


    # if dataset['Location'] == "datasets/covid_World.csv":
    #     print("\nColin: which Continent are you interested in plotting?")
    #     continent = input(username)
    #     continent_peram, continent_name = region_check(continent, dataset)
    #     print("\nColin: which Country are you interested in plotting?")
    #     country = input(username)
    #     country_peram, country_name = region_check(country, dataset)
    #     print("Colin: Please enter the start date for the time period you are interested in plotting "
    #           "Please use the format year-month-day, for example: 2020-02-29")
    #     start_date = input(username)
    #     start_date_plot = set_start_date(start_date, dataset)
    #     print("Colin: Please enter the end date for the time period you are interested in plotting "
    #           "Please use the format year-month-day, for example: 2020-02-29")
    #     end_date = input(username)
    #     end_date_plot = set_end_date(end_date, dataset)
    #     print("Colin: Which Target variable are you interested in plotting?")
    #     target = input(username)
    #     target_chosen = find_target(target, dataset)
    #
    #     decide =  decision_handler("Colin: Would you like to plot the chart with the following information,\n"
    #           "continent: " + continent + ",country: " + country + ",start-date: " + start_date + ",end-date " + end_date
    #           + ",Target: " + target,username)
    #
    #     if decide:
    #         make_line_chart(dataset,continent_peram,country_peram,target_chosen,continent_name,country_name,start_date_plot,
    #                         end_date_plot,"purple")
    #     else:
    #         print("invalid chart")
    #     return username

# scanJson method
def scanJson(field, statement):
    # lists used for choices from json file
    possibleChoice = []
    jsonChoice = []

    # for loop for traversing the various fields encapsulated inside
    # the interpretation.json file
    for key in queryTexts[field]:
        jsonChoice.append(queryTexts[field][key])
        possibleChoice.append(queryTexts[field][key]["SupportText"])

    numberChoice = query(statement, possibleChoice)
    choice = jsonChoice[numberChoice]
    return choice

# nlp user_selection function takes in user input in lower case, then checks to see if max similarity is higher or lower
# than the choices available
def user_selection(user_input, choices):
    answer1_nlp = nlp(user_input.lower())
    max_similarity = 0
    max_similarity_index = 0

    # the for loop here will check the scores of their similarity, has vector condition here is used to see if a vector
    # is not present, this will handle spelling mistakes or errors made by the user
    for i, d in enumerate(choices):
        d_nlp = nlp(d.lower())
        if answer1_nlp.has_vector:
            if answer1_nlp.similarity(d_nlp) > max_similarity:
                max_similarity = answer1_nlp.similarity(d_nlp)
                max_similarity_index = i
    if max_similarity > 0:
        return max_similarity_index
    else:
        return -1

# The query method here will handle choices the user may make that do not correspond to the datasets present
# within the library or choices the chatbot can't make out with similarity() using cosign similarity
def query(answer,choices):
    choice = -1
    while(choice == -1):
        choice = user_selection(answer, choices)
        if choice == -1:
            print("Colin: please clarify your request")
            answer = input()
    return choice

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
    print("Available features: \n")
    labelView += " |\n"
    print(labelView)

def plot_types(approach):
    if approach["Title"] == "Scatter Plot":
        plot_type = "Scatter Plot"
    elif approach["Title"] == "Bar Chart":
        plot_type = "Bar Chart"
    elif approach["Title"] == "Histogram Chart":
        plot_type = "Histogram Chart"
    elif approach["Title"] == "line Chart":
        plot_type = "line Chart"
    return plot_type

# the geopolitical_term_check method is used to identify whether a dataset is a GPE or LOC, then if the condition is met
# it will add it to a list for use with other methods
def geopolitical_term_check(text):
    doc = nlp(text)
    gpe_list = []
    for entity in doc.ents:
        #print(entity.label_)
        if entity.label_ == "GPE" or entity.label_ == "LOC":
            gpe_list.append(entity.text)
    return gpe_list

# the date_check method is similar to the geopolitical_term_check method only checks for dates
def date_check(text):
    doc = nlp(text)
    date_time = []
    for entity in doc.ents:
        if entity.label_ == "DATE":
            date_time.append(entity.text)
    return date_time

#print(geopolitical_term_check("scatter plot"))

# methods below used in testing, to see if a more dynamic method can be achieved using the graphing methods
# presented in sample_chart file
def region_check(user_input, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    if geopolitical_term_check(user_input) in df['continent'].values:
        print("In continent column")
        cont_peram = 'continent'
        cont_name = user_input
        return cont_peram,  cont_name

    elif geopolitical_term_check(user_input) in df['location'].values:
        print("In countries column")
        country_peram = 'location'
        country_name = user_input
        return country_peram, country_name
    else:
        print("Please use exact case and spelling for labels for example: Europe")

def set_start_date(user_input, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    if date_check(user_input) in df['date'].values:
        print("In start date column")
        date_start = user_input
        return  date_start
    else:
        print("Colin: Please use date format year-month-day, for example: 2020-02-29")

def set_end_date(user_input, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    if date_check(user_input) in df['date'].values:
        print("In end date column")
        date_end = user_input
        return date_end
    else:
        print("Colin: Please use date format year-month-day, for example: 2020-02-29")

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

# method for identifying user login details
def user_login(login):
    login = login + " : "
    return login

def ml_model(target, labels, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    labels = []
    target =''
    if labels in df.head(0):
        x_values = df.append(labels)
    else:
        print("Colin: The label you are looking for is not present in this dataset")

    if target in df.head(0):
        y_value = target
    else:
        print("Colin: The feature you are looking for is not present in the dataset")

def check_command(command, user):
    if "quit" in command.lower() or "exit" in command.lower():
        return not decision_handler("Colin: Are you sure you want leave the chat?", user)
    else:
        return True

def find_dataset(dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    return df

def train_model(target, labels, dataset):
    print(dataset)
    newLabels = labels
    newLabels.append(target)
    print("Labels: " + str(newLabels))
    newDataset = dataset[newLabels].copy()
    print("Dataset: ")
    print(newDataset)
    cleanDataset = newDataset.dropna()
    #cleanDataset = cleanDataset.head(50000)
    print(cleanDataset)

    X = cleanDataset[labels]
    y = cleanDataset[target]
    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = RandomForestClassifier(n_estimators=10,max_depth=6,criterion="entropy")
    print("Fitting dataset")
    clf.fit(X_train, y_train)
    print("predicting dataset")
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
dsLoc = scanJson("Datasets", "who")
dfDs = find_dataset(dsLoc)

# Function call for chatbot
chatbot()