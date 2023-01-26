import spacy as sp
import json
import pandas as pd
import Data_summary

sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

queryTexts = json.load(open('interpretation.json'))

dataset = -1
target_classification = -1
variables = -1
approach = -1


def chatbot():
    initial_setup()

def initial_setup():
    print("Colin: How may I help you with your covid related queries today?")
    statement = input()
    statement = statement.lower()

    # Getting dataset section
    content = False

    while not content:
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
        contentAnswer = input()
        if contentAnswer == 'y':
            content = True
        elif contentAnswer == 'n':
            print("Colin: Please rephrase and I will do my best to understand it")
            statement = input()
            content = False
        else:
            return

    if approach["Categorization"] == "Train":
        print("Train")
    elif approach["Categorization"] == "Plot":
        print("Plot")

    choice = decision_handler("Colin: Would you like a summary of the required Dataset?\n")
    if choice:
        Data_summary.data_summary(dataset["Location"])
    label_selection(dataset)

    print("\nColin: which label's are you interested in? \n")

    label_select = input()

def scanJson(field, statement):
    possibleChoice = []
    jsonChoice = []

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
        #print(d)
        #print(answer1_nlp.similarity(d_nlp))
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
            print("please clarify your request")
            answer = input()
    return choice

def decision_handler(question):
    summary = input(question).lower()
    while summary != "yes" and summary != 'y' \
            and summary != "no" and summary != 'n':
        print("could you please enter yes or no as your response")
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
    labelView += " |"

    print(labelView)

def plot_types(approach):
    if approach["Title"] == "Scatter Pot":
        print("Colin: Please choose which columns and rows you are interested in plotting using Scatter plot?")
    elif approach["Title"] == "Bar Chart":
        print("Colin: Please choose which columns and rows you are interested in plotting using Bar Chart?")
    elif approach["Title"] == "Histogram Chart":
        print("Colin: Please choose which columns and rows you are interested in plotting using the Histogram Chart?")

chatbot()