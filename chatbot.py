import spacy as sp
import json
import Tasks
import NLP
import plot_tasks

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
    login = input("User :")
    username = Tasks.user_login(login)
    print("Colin: Hello " + username.replace(':',',') + "Which dataset will we be working with today?")
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
            print("Goodbye " + username.replace(':',''))
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

def check_command(command, dataset,username):
    if "quit" in command.lower() or "exit" in command.lower():
        if Tasks.decision_handler("Colin: Do you wish to quit chatting " + username.replace(":",""),username):
            global done
            done = True
            return "Done"
    return NLP.phrase_match(command, dataset)

# Function call for chatbot
chatbot()


