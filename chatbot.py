import spacy as sp
import json
import pandas as pd
import Data_summary

# Nlp model chosen for the chatbot
from sample_chart import make_line_chart

sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

# Open json file
queryTexts = json.load(open('interpretation.json'))

# member variables used in conjunction with json
dataset = -1
target_classification = -1
variables = -1
approach = -1

# Chatbot main method for implementation of all its available functionality
def chatbot():
    initial_setup()

# this method is primarily used to decipher the Dataset chosen,Approach such as training or plotting
# and location of the requested Dataset from the json file named interpretation.json
def initial_setup():
    print("Colin: Hi my name is Colin, your Covid-19 chatbot, who am i talking to?")
    login = input("User :")
    login_details = login + " : "
    print("Colin: Hello " + login_details.replace(':','') + "How may I help you with your covid related queries today?")
    # statement takes in the user input in lowercase
    statement = input(login_details)
    statement = statement.lower()

    # Getting dataset section
    content = False

    while not content:
        # locates the dataset if user input matches the available dataset
        dataset = scanJson("Datasets", statement)

        # Get target classification section
        # Get approach section
        approach = scanJson("Approach", statement)

        print("Selection")
        print("-------------")
        print("Dataset: " + dataset['Title'])
        print("Approach: " + approach["Title"])
        print("Location: " + dataset['Location'])

        print("Colin: Are you happy with these details y/n")
        # checks to see if the user is happy with their selection
        # if yes chatbot will move to the next query otherwise it will ask the user
        # for further clarification
        contentAnswer = input(login_details)
        if contentAnswer == 'y':
            content = True
        elif contentAnswer == 'n':
            print("Colin: Please rephrase and I will do my best to understand it")
            statement = input(login_details)
            content = False
        else:
            return
    # the decision handler method is called here to handle the request should the user desire a summary of the
    # requested Dataset or continue on to the next query
    choice = decision_handler("Colin: Would you like a summary of the required Dataset?\n" + login_details)
    if choice:
        # Data_summary function called
        Data_summary.data_summary(dataset["Location"])
    # label_selection function called
    label_selection(dataset)

    if dataset['Location'] == "datasets/covid_World.csv":
        print("\nColin: which Continent are you interested in plotting?")
        continent = input(login_details)
        continent_peram, continent_name = region_check(continent, dataset)
        print("\nColin: which Country are you interested in plotting?")
        country = input(login_details)
        country_peram, country_name = region_check(country, dataset)
        print("Colin: Please enter the start date for the time period you are interested in plotting "
              "Please use the format year-month-day, for example: 2020-02-29")
        start_date = input(login_details)
        start_date_plot = set_start_date(start_date, dataset)
        print("Colin: Please enter the end date for the time period you are interested in plotting "
              "Please use the format year-month-day, for example: 2020-02-29")
        end_date = input(login_details)
        end_date_plot = set_end_date(end_date, dataset)
        print("Colin: Which Target variable are you interested in plotting?")
        target = input(login_details)
        target_chosen = find_target(target, dataset)
        print("Colin: Would you like to plot the chart with the following information,\n"
              "continent: " + continent + ",country: " + country + ",start-date: " + start_date + ",end-date " + end_date
              + ",Target: " + target)
        plot_chart = input(login_details)
        if decision_handler(plot_chart):
            make_line_chart(dataset,continent_peram,country_peram,target_chosen,continent_name,country_name, start_date_plot,
                            end_date_plot,"purple")

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

def user_selection(user_input, choices):
    answer1_nlp = nlp(user_input.lower())
    max_similarity = 0
    max_similarity_index = 0

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

def query(answer,choices):
    choice = -1
    while(choice == -1):
        choice = user_selection(answer, choices)
        if choice == -1:
            print("Colin: please clarify your request")
            answer = input()
    return choice

def decision_handler(question):
    summary = question.lower()
    while summary != "yes" and summary != 'y' \
            and summary != "no" and summary != 'n':
        print(summary)
        print("Colin: could you please enter yes or no as your response")
        summary = input(question).lower()
    if summary == "yes" or summary == 'y':
        return True
    else:
        return False

def label_selection(dataset):
    # Target Variable
    df = pd.read_csv(dataset['Location'], sep=',')
    labels = df.columns.values
    labelView = ""

    for i in range(0, len(labels)):
        if i % 7 == 6:
            labelView += " |\n"
        labelView += " | " + labels[i]
    print("Available labels: \n")
    labelView += " |"
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


def geopolitical_term_check(text):
    doc = nlp(text)
    gpe_list = []
    for entity in doc.ents:
        #print(entity.label_)
        if entity.label_ == "GPE" or entity.label_ == "LOC":
            gpe_list.append(entity.text)
    return gpe_list

def date_check(text):
    doc = nlp(text)
    date_time = []
    for entity in doc.ents:
        if entity.label_ == "DATE":
            date_time.append(entity.text)
    return date_time

#print(geopolitical_term_check("scatter plot"))

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
        print("Colin: please rephrase the value name by using a capital letter at the start")

def set_start_date(user_input, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    if date_check(user_input) in df['date'].values:
        print("In start date column")
        date_peram = 'date'
        date_start = user_input
        return  date_start
    else:
        print("Colin: Please use date format year-month-day, for example: 2020-02-29")

def set_end_date(user_input, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    if date_check(user_input) in df['date'].values:
        print("In end date column")
        date_peram = 'date'
        date_end = user_input
        return date_end
    else:
        print("Colin: Please use date format year-month-day, for example: 2020-02-29")

def find_target(user_input, dataset):
    df = pd.read_csv(dataset['Location'], sep=',')
    if user_input in df.head(0):
        print("found")
        target = user_input
        return target
    else:
        print("not found")
chatbot()