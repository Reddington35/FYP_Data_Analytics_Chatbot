import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import NLP

# The data_summary method provides the user with a clean representation of the data being provided by the Datasets
# they may be interested in analysing, this method provides the user with the number of Rows,columns,and shape of the
# desired Dataset they are working with while also providing the type of Data provided in each label
def data_summary(dataset):
    df = pd.read_csv(dataset['Location'],index_col=False)
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
    df = pd.read_csv(dataset,index_col=False)

def data_ml(dataset, username, request):
    df = pd.read_csv(dataset['Location'],index_col=False)
    labels = printChoices(df)

    print("Colin: Give target feature please " + username)
    choice = input(username)
    print(choice)
    # make sure we get a non-empty list from label match

    target = NLP.label_match(choice, labels)[0]
    labels = printChoices(df)
    print("Colin: Give parameters please " + username)
    choice = input(username)
    # print("Colin: Give Hyper - Parameters please " + username)
    # hp_choice = input(username)
    params = NLP.label_match(choice, labels)
    labels = []

    for label in params:
        labels.append(label)

    globals()[request['def']](target, labels, df, username)

# ML Methods
def RandomForest(target, labels, dataset,username):
    refinement = True
    print("Random Forest Classification Algorithm Selected")
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
    scaler = MinMaxScaler(feature_range=(0, 1))

    print(scaler.fit(X, y))
    model = scaler.fit((X))
    print("Max is")
    print(model.data_max_)
    print("min is")
    print(model.data_min_)
    scaled_X = model.transform(X)
    print(scaled_X)

    crit_list = ["gini","entropy"]
    cl_weight = ["balanced","balanced_subsample"]
    mx_features = ["sqrt","log2"]

    # Hyper - Parameter Default values
    user_n_estimators = 100
    user_max_depth = 2
    user_criterion = "gini"
    user_min_samples_split = 2
    user_min_samples_leaf = 1
    user_min_weight_fraction_leaf = 0.0
    user_max_features = "sqrt"
    user_max_leaf_nodes = None
    user_min_impurity_decrease = 0.0
    user_bootstrap = True
    user_oob_score = False
    user_n_jobs = None
    user_random_state = None
    user_verbose = 0
    user_warm_start = False
    user_class_weight = None
    user_ccp_alpha = 0.0
    user_max_samples = None

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3)
    while refinement:
        clf = RandomForestClassifier(n_estimators=user_n_estimators,max_depth=user_max_depth,criterion=user_criterion,
                                     min_samples_split=user_min_samples_split,min_samples_leaf=user_min_samples_leaf,
                                     min_weight_fraction_leaf=user_min_weight_fraction_leaf,max_features=user_max_features,
                                     max_leaf_nodes=user_max_leaf_nodes,min_impurity_decrease=user_min_impurity_decrease,
                                     bootstrap=user_bootstrap,oob_score=user_oob_score,n_jobs=user_n_jobs,
                                     random_state=user_random_state,verbose=user_verbose,warm_start=user_warm_start,
                                     class_weight=user_class_weight,ccp_alpha=user_ccp_alpha,max_samples=user_max_samples)
        print("Fitting dataset")
        clf.fit(X_train, y_train)
        print("predicting dataset")
        y_pred = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Recall:", metrics.recall_score(y_test, y_pred, average='micro'))
        plt.figure("Training Data")
        plt.scatter(y_test,y_pred)
        plt.title("nothing")
        plt.xlabel("Test Data")
        plt.ylabel("Predicted Data")
        plt.show()

        refine_command = input("Colin: Would you like to refine hyper-perameters ?" +"\n" + username)
        if refine_command.lower().strip() in ['quit', 'no', 'n']:
            refinement = False
        else:
            redifined_list = refine_command.split(",")
            for redifined in redifined_list:
                if "n_estimators" in redifined:
                    user_n_estimators = change_hyperperams_int("n_estimators=", redifined,user_n_estimators)
                elif "max_depth" in redifined:
                    user_max_depth = change_hyperperams_int("max_depth=", redifined,user_max_depth)
                elif "criterion" in redifined:
                    for crit in crit_list:
                        if crit in redifined:
                            user_criterion = change_hyperperams_String("criterion=",redifined,user_criterion)
                elif "min_samples_split" in redifined:
                    user_min_samples_split = change_hyperperams_int("min_samples_split=",redifined,user_min_samples_split)
                elif "min_samples_leaf" in redifined:
                    user_min_samples_leaf = change_hyperperams_int("min_samples_leaf=",redifined,user_min_samples_leaf)
                elif "min_weight_fraction_leaf" in redifined:
                    user_min_weight_fraction_leaf = change_hyperperams_Float("min_weight_fraction_leaf=",redifined,
                                                                             user_min_weight_fraction_leaf)
                elif "max_features" in redifined:
                    for mxf in mx_features:
                        if mxf in redifined:
                            user_max_features = change_hyperperams_String("max_features=",redifined,user_max_features)
                elif "max_leaf_nodes" in redifined:
                    user_max_leaf_nodes = change_hyperperams_int("max_leaf_nodes=",redifined,user_max_leaf_nodes)
                elif "min_impurity_decrease" in redifined:
                    user_min_impurity_decrease = change_hyperperams_Float("min_impurity_decrease=",redifined,
                                                                          user_min_impurity_decrease)
                elif "bootstrap" in redifined:
                    user_bootstrap = change_hyperperams_bool("bootstrap=",redifined,user_bootstrap)
                elif "oob_score" in redifined:
                    user_oob_score = change_hyperperams_bool("oob_score=",redifined,user_oob_score)
                elif "n_jobs" in redifined:
                    user_n_jobs = change_hyperperams_int("n_jobs=",redifined,user_n_jobs)
                elif "random_state" in redifined:
                    user_random_state = change_hyperperams_int("random_state=",redifined,user_random_state)
                elif "verbose" in redifined:
                    user_verbose = change_hyperperams_int("verbose=",redifined,user_verbose)
                elif "warm_start" in redifined:
                    user_warm_start = change_hyperperams_bool("warm_start=",redifined,user_warm_start)
                elif "class_weight" in redifined:
                    for weight in cl_weight:
                        if weight in redifined:
                            user_class_weight = change_hyperperams_String("class_weight=",redifined,user_class_weight)
                elif "ccp_alpha" in redifined:
                    user_ccp_alpha = change_hyperperams_String("ccp_alpha=",redifined,user_ccp_alpha)
                elif "max_samples" in redifined:
                    user_max_samples = change_hyperperams_Float("max_samples=",redifined,user_max_samples)
                elif "help" in redifined:
                    helper()

def NaiveBayes(target, labels, dataset):
    print("Naive Bayes Classification Algorithm Selected")
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
    scaler = MinMaxScaler(feature_range=(0, 1))

    print(scaler.fit(X, y))
    model = scaler.fit((X))
    print("Max is")
    print(model.data_max_)
    print("min is")
    print(model.data_min_)
    scaled_X = model.transform(X)
    print(scaled_X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = GaussianNB()
    print("Fitting dataset")
    clf.fit(X_train, y_train)
    print("predicting dataset")
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def SupportVectorMachine(target, labels, dataset):
    print("SVM Classification Algorithm Selected")
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
    scaler = MinMaxScaler(feature_range=(0, 1))

    print(scaler.fit(X, y))
    model = scaler.fit((X))
    print("Max is")
    print(model.data_max_)
    print("min is")
    print(model.data_min_)
    scaled_X = model.transform(X)
    print(scaled_X)

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
    df = pd.read_csv(dataset['Location'], index_col=False)
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
    df = pd.read_csv(dataset['Location'],index_col=False)
    return df

# decision_handler method is used to manage basic yes or no patterns in questions
# no nlp is used here just takes in a question and user input as parameters and manages the response the user provides
# if no yes/no response is given the method will ask the user to simply respond yes/no
def decision_handler(question,username):
    print(question)
    summary = input(username)
    while summary.lower() != "yes" and summary.lower() != 'y' \
            and summary.lower() != "no" and summary.lower() != 'n':
        print("Colin: could you please enter yes or no as your response")
        print(question.lower())
        summary = input(username)
    if summary.lower() == "yes" or summary.lower() == 'y':
        print("Colin: Thank You " + username.replace(':','').strip())
        return True
    else:
        return False

# find_target method searches the head of the dataset (the Features) and checks to see if their a match
# with the user input
def find_target(user_input, dataset):
    df = pd.read_csv(dataset['Location'], index_col=False)
    if user_input in df.head(0):
        print("found")
        target = user_input
        return target
    else:
        print("not found")

def change_hyperperams_int(hyper_peram,refine_command,Default_value):
    result = Default_value
    if hyper_peram in refine_command:
        if "=" in refine_command:
            result =  int(refine_command[refine_command.rfind("=") + 1:])
        else:
            result = int(refine_command[refine_command.rfind(" ") + 1:])
    print("Colin: Changed "+ hyper_peram.replace('=','').strip() + " to " + str(result))
    return result

def change_hyperperams_String(hyper_peram,refine_command,Default_value):
    result = Default_value
    if hyper_peram in refine_command:
        if "=" in refine_command:
            result = refine_command[refine_command.rfind("=") + 1:]
        else:
            result = refine_command[refine_command.rfind(" ") + 1:]
    print("Colin: Changed " + hyper_peram.replace('=','').strip() + " to " + result)
    return result

def change_hyperperams_Float(hyper_peram,refine_command,Default_value):
    result = Default_value
    if hyper_peram in refine_command:
        if "=" in refine_command:
            result = float(refine_command[refine_command.rfind("=") + 1:])
        else:
            result = float(refine_command[refine_command.rfind(" ") + 1:])
    print("Colin: Changed " + hyper_peram.replace('=','').strip() + " to " + str(result))
    return result

def change_hyperperams_bool(hyper_peram,refine_command,Default_value):
    commandResult = Default_value
    if hyper_peram in refine_command:
        if "=" in refine_command:
            commandResult = refine_command[refine_command.rfind("=") + 1:].lower() in ['true']

        else:
            commandResult = refine_command[refine_command.rfind(" ") + 1:].lower() in ['true']

    print("Colin: Changed " + hyper_peram.replace('=','').strip() + " to " + str(commandResult))
    return commandResult

def helper():
    print("- Criterion takes the values 'gini' or 'entropy' for example  type: criterion=gini"
          ", as the command to change this hyper-parameters\n- n_estimators takes an int value, "
          "for example type: n_estimators=10"
          ", as the command to change this hyper-parameters\n- max_depth takes an int value for example  type: criterion=gini"
          ", as the command to change this hyper-parameters\n")