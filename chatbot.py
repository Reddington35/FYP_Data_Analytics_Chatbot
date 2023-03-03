from os.path import isfile
from datetime import datetime
import spacy as sp
import json
import Tasks
import NLP
import plot_tasks
import pickle

# Global list to be used with record feedback
user_details = []
# Global list to be used with record conversation
conversation = []
# Open json file
queryTexts = json.load(open("interpretation.json"))

# member variables used in conjunction with json
dataset = -1
target_classification = -1
variables = -1
approach = -1
done = False

# Nlp model chosen for the chatbot
sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

# This is the main function of the chatbot and will handle any conversational functionality
# this method is primarily used to decipher the Dataset chosen,Approach such as training or plotting
# and location of the requested Dataset from the json file named interpretation.json
# and operate the chatbot with the functions written to improve its mechanics
def chatbot():
    print("Colin: Hi my name is Colin, your Covid-19 chatbot, who am i talking to?")
    login = input("User:")
    username = user_login(login)
    path = "Users/" + username.replace(':','').strip() + ".pkl"
    if isfile(path):
        preferences = Tasks.decision_handler("Colin: Hello again " + username.replace(':','').strip() +
                                            " Would you like to see a transcript of our previous conversation?",username)
        if preferences:
            print("Previous Conversation with User " + username.replace(':','').strip()
                  + "\n****************************" )
            deserialise_user(username)
    print("Colin: " + "So " + username.replace(':','').strip() + " Which dataset will we be working with today?")

    # statement takes in the user input in lowercase
    statement = input(username)
    statement = statement.lower()
    dataset = Tasks.load_dataset(statement,queryTexts,username)

    # Get Dataset
    # Basic Instructions on the services available by the chatbot
    print("Colin: I am a covid-19 chatbot, however I can only provide "
              "help on topics such as:\n - Data summary of the Dataset\n - Graph the Dataset\n - Build a ml model of the"
              " dataset using (Random Forrest, Naive Bayes or SVM)\n - Display the available Features")
    # While loop that checks to see if the user wants to quit the application
    # or choose from four distinct services available
    while not done:
        print("Colin: which task would you like me to perform?")
        task = input(username)
        # Are we changing Dataset or running Approach?
        request = check_command(task, queryTexts["Approach"],username)

        if request == "Done":
            print("Colin: signing you out now " + username.replace(':',''))
        else:
            if request["Categorization"] == "Change_Dataset":
                dataset = Tasks.load_dataset(task, queryTexts, username)

            elif request["Categorization"] == "Plot":
                #Tasks.data_plot(dataset,task, request)
                if request["Title"] == "Line Chart":
                    #NLP.world_dataset_region(dataset, username)
                    plot_tasks.dynamic_line_chart(dataset,task,task,username)
                elif request["Title"] == "Bar Chart":
                    plot_tasks.dynamic_bar_chart(dataset,task,task,username)
                elif request["Title"] == "Scatter Plot":
                    plot_tasks.dynamic_scatter_chart(dataset,task,task,username)
                elif request["Title"] == "Histogram Chart":
                    plot_tasks.dynamic_histogram(dataset,task,username)

            elif request["Categorization"] == "Train":
                Tasks.data_ml(dataset, username, request)

            elif request["Categorization"] == "Display":
                Tasks.data_summary(dataset)

            elif request["Categorization"] == "Features":
                Tasks.feature_selection(dataset)
    customer_feedback(username)

def check_command(command, dataset,username):
    if "quit" in command.lower() or "exit" in command.lower():
        if Tasks.decision_handler("Colin: Do you wish to quit chatting " + username.replace(":","").strip(),username):
            global done
            done = True
            return "Done"
    return NLP.phrase_match(command, dataset)

def customer_feedback(username):
    user_complete = Tasks.decision_handler("Colin: Before I complete signing you out of the chat,"
                           " Would you mind completing some short feedback questions"
                           " on your user experience " + username.replace(":","").strip() + "?",username)
    if not user_complete:
        print("Colin: Thank You " + username.replace(':','').strip() + " and have a good day")
    if user_complete:
        print("Colin: Excellent")
        print("Welcome to customer feedback\n*****************************")
        record_chat("Colin: Did you have any trouble with the Machine Learning feature of this application,"
                    "if so which feature did you have trouble using?",username)
        record_chat("Colin: Did you have any trouble with the Dynamic Graphing feature of this application,"
                    "if so which feature did you have trouble using?",username)
        record_chat("Colin: Did you find me efficient,in how I performed the tasks asked of me, if not where "
                    "in your opinion could I improve?",username)
        print("Colin: Thank You " + username.replace(':','').strip() + " and have a good day")

def record_chat(colin, username):
    print(colin)
    user = input(username)
    add_user(username)
    user_details.append(colin)
    user_details.append(username + user)
    with open('Users/'+ username.replace(':','').strip() +'.pkl', 'wb') as f:
        pickle.dump(user_details, f)

def deserialise_user(username):
    with open('Users/'+ username.replace(':','').strip() +'.pkl', 'rb') as f:
        users = add_user(username)
        if username in users:
            print_feedback_chat = pickle.load(f)
            for p in print_feedback_chat:
                print(p)

# method for identifying user login details
def user_login(login):
    login = login + " : "
    return login

def add_user(username):
    users = []
    users.append(username)
    return users

def print_and_log(message):
    start_time = datetime.now()
    if len(message > 0):
        conversation.append(message)
    end_date = datetime.now()
    print(message)

def input_and_log(message,username):
    start_date = datetime.now()
    if len(message > 0):
        conversation.append(message)
    user_input = input(username)
    if len(user_input > 0):
        conversation.append(user_input)
    end_date = datetime.now()
    return user_input

# Function call for chatbot
chatbot()



