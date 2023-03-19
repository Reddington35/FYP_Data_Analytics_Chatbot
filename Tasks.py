import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, \
    precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import NLP
import warnings

warnings.filterwarnings('ignore')

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

    target = NLP.label_match(choice, labels,username)[0]
    labels = printChoices(df)
    print("Colin: Give parameters please " + username)
    choice = input(username)
    # print("Colin: Give Hyper - Parameters please " + username)
    # hp_choice = input(username)
    params = NLP.label_match(choice, labels,username)
    labels = []

    for label in params:
        labels.append(label)

    globals()[request['def']](target, labels, df, username)

# ML Methods
def RandomForestClassification(target, labels, dataset,username):
    try:
        refinement = True
        #print(dataset)
        newLabels = labels
        newLabels.append(target)
        #print("Labels: " + str(newLabels))
        newDataset = dataset[newLabels].copy()
        #print("Dataset: ")
        #print(newDataset)
        cleanDataset = newDataset.dropna()
        # cleanDataset = cleanDataset.head(50000)
        #print(cleanDataset)

        X = cleanDataset[labels]
        y = cleanDataset[target]
        #print(X)
        #print(y)

        # use StandardScaler to normalise and scale the data
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        #print(scaled_X)

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
            print("Colin: Fitting dataset using Random Forest Classifier")
            clf.fit(X_train, y_train)
            print("Colin: predicting dataset using Random Forest Classifier")
            y_pred = clf.predict(X_test)
            print("Colin: Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))
            #pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predicted'], margins =True)
            #print("Classification report:\n", classification_report(y_test,y_pred))
            #y_pred_proba = clf.predict_proba(X_test)[:, 1]
            #roc_curve(y_test,y_pred)
            #print("Area under Curve score", roc_auc_score(y_test, y_pred))
            # precision, recall, thresholds = precision_recall_curve(y_test,y_pred)
            # print(thresholds)
            #
            # # Area under curve
            # auc_pred = auc(recall, precision)
            # print(auc_pred)
            print("Colin: f1 score = ",metrics.f1_score(y_test, y_test, average='micro',labels=np.unique(y_pred)))
            print("Colin: Precision =",metrics.precision_score(y_test, y_pred, average='weighted',
                                                               labels=np.unique(y_pred)))
            print("Colin: Accuracy:", metrics.accuracy_score(y_test, y_pred))
            print("Colin: Recall:", metrics.recall_score(y_test, y_pred,average='weighted',labels=np.unique(y_pred)))
            plt.figure("Machine Learning Data")
            plt.scatter(y_test,y_pred)
            slope, intercept, r_value, p_value, std_err = linregress(y_test,y_pred)
            plt.plot(y_test, intercept + slope * y_test, 'r', label='fitted line')
            plt.title("Actual vs Predicted Data")
            plt.xlabel("Actual Data",color='purple')
            plt.ylabel("Predicted Data",color='green')
            plt.show()

            refine_command = input("Colin: Would you like to refine hyper-parameters ?" +"\n" + username)
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
    except ValueError:
       print("Colin: oops there was an issue training these Features, sorry " + username.replace(':', '').strip())

def NaiveBayes(target, labels, dataset,username):
    try:
        #print(dataset)
        newLabels = labels
        newLabels.append(target)
        #print("Labels: " + str(newLabels))
        newDataset = dataset[newLabels].copy()
        #print("Dataset: ")
        #print(newDataset)
        cleanDataset = newDataset.dropna()
        # cleanDataset = cleanDataset.head(50000)
        #print(cleanDataset)

        X = cleanDataset[labels]
        y = cleanDataset[target]
        #print(X)
        #print(y)

        # use StandardScaler to normalise and scale the data
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        #print(scaled_X)

        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3)

        clf = GaussianNB()
        print("Colin: Fitting dataset using Naive Bayes GaussianNB Classifier")
        clf.fit(X_train, y_train)
        print("Colin: predicting dataset using Naive Bayes GaussianNB Classifier")
        y_pred = clf.predict(X_test)
        print("Colin: Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
        print("Colin: f1 score = ", metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
        print("Colin: Precision =", metrics.precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
        print("Colin: Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Colin: Recall:", metrics.recall_score(y_test, y_pred, average='macro'))
        plt.figure("Machine Learning Data")
        plt.scatter(y_test, y_pred)
        slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
        plt.plot(y_test, intercept + slope * y_test, 'r', label='fitted line')
        plt.title("Actual vs Predicted Data")
        plt.xlabel("Actual Data")
        plt.ylabel("Predicted Data")
        plt.show()
    except ValueError:
        print("Colin: oops there was an issue training these Features, sorry " + username.replace(':','').strip())

def RandomForestRegression(target, labels, dataset,username):
    try:
        refinement = True
        #print(dataset)
        newLabels = labels
        newLabels.append(target)
        #print("Labels: " + str(newLabels))
        newDataset = dataset[newLabels].copy()
        #print("Dataset: ")
        #print(newDataset)
        cleanDataset = newDataset.dropna()
        # cleanDataset = cleanDataset.head(50000)
        #print(cleanDataset)

        X = cleanDataset[labels]
        y = cleanDataset[target]
        #print(X)
        #print(y)

        crit_list = ["squared_error","absolute_error","friedman_mse","poisson"]
        mx_features = ["sqrt", "log2"]

        # Hyper - Parameter Default values
        user_n_estimators = 100
        user_max_depth = None
        user_criterion = "squared_error"
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
        user_ccp_alpha = 0.0
        user_max_samples = None
        while refinement:
            clf = RandomForestRegressor(n_estimators=user_n_estimators,max_depth=user_max_depth,criterion=user_criterion,
                                         min_samples_split=user_min_samples_split,min_samples_leaf=user_min_samples_leaf,
                                         min_weight_fraction_leaf=user_min_weight_fraction_leaf,max_features=user_max_features,
                                         max_leaf_nodes=user_max_leaf_nodes,min_impurity_decrease=user_min_impurity_decrease,
                                         bootstrap=user_bootstrap,oob_score=user_oob_score,n_jobs=user_n_jobs,
                                         random_state=user_random_state,verbose=user_verbose,warm_start=user_warm_start,
                                         ccp_alpha=user_ccp_alpha,max_samples=user_max_samples)
            kf = KFold(n_splits=10,random_state=10, shuffle=True)
            kf.get_n_splits(X)
            count = 0
            for train_index,test_index in kf.split(X):
               count += 1
               X_train,X_test = X.iloc[train_index],X.iloc[test_index]
               y_train,y_test = y.iloc[train_index],y.iloc[test_index]
               print("Colin: Fitting dataset using Random Forest Regressor fold ",count)
               clf.fit(X_train,y_train)
               print("Colin: predicting dataset using Random Forest Regressor fold ",count)
               y_pred = clf.predict(X_test)
            print("Colin: Mean Squared Error = ",metrics.mean_squared_error(y_test,y_pred))
            print("Colin: Root Mean Squared Error = ", metrics.mean_squared_error(y_test, y_pred,squared=False))
            print("Colin: Mean Absolute Error = ",metrics.mean_absolute_error(y_test,y_pred))
            print("Colin: r2 score = ", metrics.r2_score(y_test,y_pred))
            plt.figure("Machine Learning Data")
            plt.scatter(y_test, y_pred)
            slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
            plt.plot(y_test, intercept + slope * y_test, 'r', label='fitted line')
            plt.title("Actual vs Predicted Data")
            plt.xlabel("Actual Data")
            plt.ylabel("Predicted Data")
            plt.show()

            refine_command = input("Colin: Would you like to refine hyper-perameters ?" + "\n" + username)
            if refine_command.lower().strip() in ['quit', 'no', 'n']:
                refinement = False
            else:
                redifined_list = refine_command.split(",")
                for redifined in redifined_list:
                    if "n_estimators" in redifined:
                        user_n_estimators = change_hyperperams_int("n_estimators=", redifined, user_n_estimators)
                    elif "max_depth" in redifined:
                        user_max_depth = change_hyperperams_int("max_depth=", redifined, user_max_depth)
                    elif "criterion" in redifined:
                        for crit in crit_list:
                            if crit in redifined:
                                user_criterion = change_hyperperams_String("criterion=", redifined, user_criterion)
                    elif "min_samples_split" in redifined:
                        user_min_samples_split = change_hyperperams_int("min_samples_split=", redifined,
                                                                        user_min_samples_split)
                    elif "min_samples_leaf" in redifined:
                        user_min_samples_leaf = change_hyperperams_int("min_samples_leaf=", redifined,
                                                                       user_min_samples_leaf)
                    elif "min_weight_fraction_leaf" in redifined:
                        user_min_weight_fraction_leaf = change_hyperperams_Float("min_weight_fraction_leaf=", redifined,
                                                                                 user_min_weight_fraction_leaf)
                    elif "max_features" in redifined:
                        for mxf in mx_features:
                            if mxf in redifined:
                                user_max_features = change_hyperperams_String("max_features=", redifined, user_max_features)
                    elif "max_leaf_nodes" in redifined:
                        user_max_leaf_nodes = change_hyperperams_int("max_leaf_nodes=", redifined, user_max_leaf_nodes)
                    elif "min_impurity_decrease" in redifined:
                        user_min_impurity_decrease = change_hyperperams_Float("min_impurity_decrease=", redifined,
                                                                              user_min_impurity_decrease)
                    elif "bootstrap" in redifined:
                        user_bootstrap = change_hyperperams_bool("bootstrap=", redifined, user_bootstrap)
                    elif "oob_score" in redifined:
                        user_oob_score = change_hyperperams_bool("oob_score=", redifined, user_oob_score)
                    elif "n_jobs" in redifined:
                        user_n_jobs = change_hyperperams_int("n_jobs=", redifined, user_n_jobs)
                    elif "random_state" in redifined:
                        user_random_state = change_hyperperams_int("random_state=", redifined, user_random_state)
                    elif "verbose" in redifined:
                        user_verbose = change_hyperperams_int("verbose=", redifined, user_verbose)
                    elif "warm_start" in redifined:
                        user_warm_start = change_hyperperams_bool("warm_start=", redifined, user_warm_start)
                    elif "ccp_alpha" in redifined:
                        user_ccp_alpha = change_hyperperams_String("ccp_alpha=", redifined, user_ccp_alpha)
                    elif "max_samples" in redifined:
                        user_max_samples = change_hyperperams_Float("max_samples=", redifined, user_max_samples)
                    elif "help" in redifined:
                        helper()
    except ValueError:
        print("Colin: oops there was an issue training these Features, sorry " + username.replace(':','').strip())

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
    print("Dataset: " + dataset['Title'] + "\n")
    print("Available Features: \n")
    labelView += " |\n"
    print(labelView)

def load_dataset(statement,queryTexts,username):
    # Getting dataset section
    content = False
    while not content:
        # locates the dataset if user input matches the available dataset
        dataset = NLP.phrase_match(statement, queryTexts["Datasets"],username)

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
        while not contentAnswer:
            print("Colin: " + "No problem " + username.replace(':','').strip() + " Which dataset would you like to use?,\n"
                  "- covid_Eu.csv\n- covid_Ireland.csv\n- covid_World.csv\n- WHO_covid.csv")
            statement = input(username)
            # locates the dataset if user input matches the available dataset
            dataset = NLP.phrase_match(statement, queryTexts["Datasets"],username)
            # Get target classification section
            print("Selection")
            print("-------------")
            print("Dataset: " + dataset['Title'])
            print("Location: " + dataset['Location'])

            responce = decision_handler("Colin: Are you happy with these details y/n",username)
            if responce:
                contentAnswer = True
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
            result = int(refine_command[refine_command.rfind("=") + 1:])
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
