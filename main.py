import pandas as pd
import requests
import numpy as np
import spacy as sp
import sklearn
from sklearn.ensemble import RandomForestClassifier

sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

datasets = ["COVID-19 HSE Daily Vaccination Figures Ireland 2021/04/07 - 2022/12/04",
            "Data on COVID-19 vaccination in the EU/EEA 2020-W53 - 2022-W52","World covid-19 vaccination dataset"]

def chatbot():
    print("Hi this is colin, what dataset will you be working with today?")

    # Find the most similar
    answer1 = input()
    answer1_nlp = nlp(answer1)

    max_similarity = 0
    max_similarity_index = 0
    for i,d in enumerate(datasets):
        d_nlp = nlp(d)

        if d_nlp.similarity(answer1_nlp) > max_similarity:
            max_similarity = d_nlp.similarity(answer1_nlp)
            max_similarity_index = i
    if max_similarity > 0:
        print("found dataset" + datasets[max_similarity_index] + ". Do you want to use this dataset?")
        answer2 = input()

        if answer2 == "yes":
            print("What data analysis task would you like performed ?")
            answer3 = input()
            print("machine learning beep,beep")

chatbot()
















