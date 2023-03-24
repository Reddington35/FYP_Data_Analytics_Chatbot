import os
from os.path import isfile, join
from datetime import datetime
import spacy as sp
import json
import Tasks
import NLP
import plot_tasks
import pickle
import glob

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
    print("**************************************************\n"
          "********   ********    *         **   ** *       *\n"
          "*         **      **   *         **   **   *     *\n"
          "*         **      **   *         **   **     *   *\n"
          "*         **      **   *         **   **       * *\n"
          "********   ********    ********  **   **         *\n"
          "**************************************************\n")
    Tasks.print_and_log("Colin: Hi my name is Colin, your Covid-19 chatbot, who am i talking to?",conversation)
    login = Tasks.input_and_log("User:",conversation)
    username = user_login(login[:20])
    path = '/Users/jamesreddington/PycharmProjects/fyp_chatbot_colin/Users/'
    files = [filename for filename in os.listdir(path) if filename.startswith(username.replace(':','').strip())]
    if len(files) > 0:
        preferences = Tasks.decision_handler("Colin: Hello again " + username.replace(':', '').strip() +
                                             " Would you like to see a transcript of our last conversation?",username)
        if preferences:
            print("Previous Conversation with User " + username.replace(':', '').strip()
                  + "\n****************************")
            valid_file = str(username.replace(':','').strip())
            directory_path = r'/Users/jamesreddington/PycharmProjects/fyp_chatbot_colin/Users/'
            file_type = r'/*' + valid_file + '*pkl'
            filing = glob.glob(directory_path + file_type)
            sorted_file = max(filing, key=os.path.getctime)
            deserialise_user(sorted_file)

    Tasks.print_and_log("Colin: " + "So " + username.replace(':','').strip() + " Which dataset will we be working with today?\n"
          "- covid_Eu.csv\n- covid_Ireland.csv\n- covid_World.csv\n- WHO_covid.csv\n- Custom.csv\n"
                                                                               "- drug_prediction",conversation)

    # statement takes in the user input in lowercase
    statement = Tasks.input_and_log(username,conversation)
    statement = statement.lower()
    dataset = Tasks.load_dataset(statement,queryTexts,username,conversation)

    # Get Dataset
    # Basic Instructions on the services available by the chatbot
    Tasks.print_and_log("Colin: I am a covid-19 chatbot, however I can only provide "
              "help on topics such as:\n - Data summary of the Dataset\n - Graph the Dataset\n - Build a ml model of the"
              " dataset using (Random Forrest Classification, Naive Bayes or Random Forest Regression"
              "\n - Display the available Features",conversation)
    # While loop that checks to see if the user wants to quit the application
    # or choose from four distinct services available
    while not done:
        Tasks.print_and_log("Colin: which task would you like me to perform?",conversation)
        task = Tasks.input_and_log(username,conversation)
        # Are we changing Dataset or running Approach?
        request = check_command(task, queryTexts["Approach"],username)

        if request == "Done":
            Tasks.print_and_log("Colin: No Problem " + username.replace(':',''),conversation)
        else:
            if request["Categorization"] == "Change_Dataset":
                dataset = Tasks.load_dataset(task, queryTexts, username,conversation)

            elif request["Categorization"] == "Plot":
                if request["Title"] == "Line Chart":
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
                Tasks.data_summary(dataset,conversation)

            elif request["Categorization"] == "Features":
                Tasks.feature_selection(dataset)
    customer_feedback(username)
    # save conversation to pickle file
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conversation.append(timestamp + "\nend of conversation\n")
    with open(path + username.replace(':','').strip() + timestamp + ".pkl", 'wb') as f:
        pickle.dump(conversation, f)

def check_command(command, dataset,username):
    if "quit" in command.lower() or "exit" in command.lower():
        if Tasks.decision_handler("Colin: Do you wish to quit chatting " + username.replace(":","").strip(),username):
            global done
            done = True
        else:
            done = False
        return "Done"
    return NLP.phrase_match(command, dataset,username)

def customer_feedback(username):
    user_complete = Tasks.decision_handler("Colin: Before I complete signing you out of the chat,"
                           " Would you please complete some short feedback questions"
                           " on your user experience " + username.replace(":","").strip() + "?",username)
    if not user_complete:
        Tasks.print_and_log("Colin: Thank You " + username.replace(':','').strip() + " and have a good day",conversation)
    if user_complete:
        Tasks.print_and_log("Welcome to customer feedback\n*****************************",conversation)
        Tasks.print_and_log("Colin: Did you have any trouble with the Machine Learning feature of this application,"
                    "if so which feature did you have trouble using?",conversation)
        Tasks.input_and_log(username,conversation)
        Tasks.print_and_log("Colin: Did you have any trouble with the Dynamic Graphing feature of this application,"
                    "if so which feature did you have trouble using?",conversation)
        Tasks.input_and_log(username,conversation)
        Tasks.print_and_log("Colin: Did you find me efficient,in how I performed the tasks asked of me, if not where "
                    "in your opinion could I improve?",conversation)
        Tasks.input_and_log(username,conversation)
        Tasks.print_and_log("Colin: Thank You " + username.replace(':','').strip() + " and have a good day",conversation)

def deserialise_user(path):
    with open(path, 'rb') as f:
        print_chat = pickle.load(f)
        for p in print_chat:
            print(p)

# method for identifying user login details
def user_login(login):
    login = login + ":"
    return login

def add_user(username):
    users = []
    users.append(username)
    return users

# Function call for chatbot
chatbot()



