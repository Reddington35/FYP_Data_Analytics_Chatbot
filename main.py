import pandas as pd
import requests
import numpy as np
import spacy as sp
import sklearn
from sklearn.ensemble import RandomForestClassifier
#import rand_forrest_model

sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

dataset_locations = {
    0: "datasets/covid_Ireland.csv",
    1: "datasets/covid_EU.csv",
    2: "datasets/covid_World.csv"
}

ml_models = {
    0: ["Naive Bayes","Random Forrest","Support Vector Machine"],
    1: ["Naive Bayes","Random Forrest","Support Vector Machine"],
    2: ["Naive Bayes","Random Forrest","Support Vector Machine"]
}

ml_location = {
    0: "Naive Bayes",
    1: "Random Forrest",
    2: "Support Vector Machine"
}

ml_hyper_perams = {

}


datasets = ["COVID-19 HSE Daily Vaccination Figures Ireland 2021/04/07 - 2022/12/04",
            "Data on COVID-19 vaccination in the EU/EEA 2020-W53 - 2022-W52 eu europe european union", "World covid-19 vaccination dataset"]

def user_selection(user_input,choices):
    answer1_nlp = nlp(user_input)

    max_similarity = 0
    max_similarity_index = 0
    for i, d in enumerate(choices):
        d_nlp = nlp(d)

        if d_nlp.similarity(answer1_nlp) > max_similarity:
            max_similarity = d_nlp.similarity(answer1_nlp)
            max_similarity_index = i

    if max_similarity > 0:
        return max_similarity_index
    else:
        return -1


# chatbot function
def chatbot():
    print("Colin: Hi this is colin, what dataset will you be working with today?")

    # Find the most similar
    answer1 = input()
    dataset_choice = user_selection(answer1,datasets)
    if dataset_choice != -1:
        print("Colin: dataset found, " + dataset_locations[dataset_choice] + ". Do you wish to use this dataset?")
        answer2 = input()
        if answer2 == "yes":
            print("Colin: What data analysis task would you like performed ?")
            answer3 = input()
            mlIndex = user_selection(answer3, ml_models[dataset_choice])

            print("Colin: found Machine Learning Model " + ml_location[mlIndex] + ", Do you want to use this model?")
            if answer3 == "yes":
                print("Colin: Which Target variable are you interested in for covid 19 Dataset")
                answer4 = input()

chatbot()
















