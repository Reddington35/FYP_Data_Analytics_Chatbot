import pandas as pd
import requests
import numpy as np
import spacy as sp
import sklearn
from sklearn.ensemble import RandomForestClassifier

import Data_summary

# NLP model set to prefer the use of Graphics Processing unit, using the en_core_web_lg english pipeline,
# for a higher accuracy evaluation
from Visualisation import visualisation_task

sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

# Dictionary which displays th locations of the csv Datasets associated with this project
dataset_locations = {
    0: "datasets/covid_Ireland.csv",
    1: "datasets/covid_EU.csv",
    2: "datasets/covid_World.csv"
}

# list containing the number of tasks the user can ask to be performed on the Dataset of choice
tasks_performed = ["Machine Learning(Ml)","train model","Visualisation","Plot chart"
                  ,"Scatter plot","Bar chart","pie chart","Histogram"]

# List detailing the associated ML classifiers associated with this project
ml_models = ["naive bayes", "random forrest", "support vector machine(SVM)"]

# location of each classifier for use in similarity comparison
ml_location = {
    0: "Naive Bayes",
    1: "Random Forrest",
    2: "Support Vector Machine"
}

# Dictionary storing all available labels associated with each of the Datasets provided
dataset_labels = {
    0: ["X","Y","VaccinationDate","VaccineText","Dose1","Dose2",
        "SingleDose","Dose1Cum","Dose2Cum","SingleDoseCum","PartiallyVacc","FullyVacc","PartialPercent","FullyPercent","ObjectId"],

    1: ["iso_code","continent","location","date","total_cases","new_cases","new_cases_smoothed","total_deaths","new_deaths",
             "new_deaths_smoothed","total_cases_per_million","new_cases_per_million","new_cases_smoothed_per_million","total_deaths_per_million",
             "new_deaths_per_million","new_deaths_smoothed_per_million","reproduction_rate,icu_patients","icu_patients_per_million",
             "hosp_patients","hosp_patients_per_million","weekly_icu_admissions","weekly_icu_admissions_per_million","weekly_hosp_admissions",
             "weekly_hosp_admissions_per_million","total_tests","new_tests","total_tests_per_thousand","new_tests_per_thousand,new_tests_smoothed",
             "new_tests_smoothed_per_thousand,positive_rate","tests_per_case,tests_units","total_vaccinations","people_vaccinated",
             "people_fully_vaccinated","total_boosters","new_vaccinations","new_vaccinations_smoothed","total_vaccinations_per_hundred",
             "people_vaccinated_per_hundred","people_fully_vaccinated_per_hundred","total_boosters_per_hundred","new_vaccinations_smoothed_per_million",
             "new_people_vaccinated_smoothed","new_people_vaccinated_smoothed_per_hundred","stringency_index","population_density","median_age",
             "aged_65_older","aged_70_older","gdp_per_capita","extreme_poverty","cardiovasc_death_rate","diabetes_prevalence","female_smokers","male_smokers","handwashing_facilities",
             "hospital_beds_per_thousand","life_expectancy","human_development_index","population","excess_mortality_cumulative_absolute","excess_mortality_cumulative","excess_mortality",
             "excess_mortality_cumulative_per_million""new_cases","new_cases_smoothed","total_deaths","new_deaths",
             "new_deaths_smoothed","total_cases_per_million","new_cases_per_million","new_cases_smoothed_per_million","total_deaths_per_million",
             "new_deaths_per_million","new_deaths_smoothed_per_million","reproduction_rate,icu_patients","icu_patients_per_million",
             "hosp_patients","hosp_patients_per_million","weekly_icu_admissions","weekly_icu_admissions_per_million","weekly_hosp_admissions",
             "weekly_hosp_admissions_per_million","total_tests","new_tests","total_tests_per_thousand","new_tests_per_thousand,new_tests_smoothed",
             "new_tests_smoothed_per_thousand,positive_rate","tests_per_case,tests_units","total_vaccinations","people_vaccinated",
             "people_fully_vaccinated","total_boosters","new_vaccinations","new_vaccinations_smoothed","total_vaccinations_per_hundred",
             "people_vaccinated_per_hundred","people_fully_vaccinated_per_hundred","total_boosters_per_hundred","new_vaccinations_smoothed_per_million",
             "new_people_vaccinated_smoothed","new_people_vaccinated_smoothed_per_hundred","stringency_index","population_density","median_age",
             "aged_65_older","aged_70_older","gdp_per_capita","extreme_poverty","cardiovasc_death_rate","diabetes_prevalence","female_smokers","male_smokers","handwashing_facilities",
             "hospital_beds_per_thousand","life_expectancy","human_development_index","population","excess_mortality_cumulative_absolute","excess_mortality_cumulative","excess_mortality",
             "excess_mortality_cumulative_per_million"],

    2: ["YearWeekISO","ReportingCountry","Denominator","NumberDosesReceived","NumberDosesExported","FirstDose",
             "FirstDoseRefused","SecondDose","DoseAdditional1","DoseAdditional2","DoseAdditional3","UnknownDose","Region",
             "TargetGroup","Vaccine","Population"]
}

datasets = ["COVID-19 HSE Daily Vaccination Figures Ireland 2021/04/07 - 2022/12/04",
            "Data on COVID-19 vaccination in the EU/EEA 2020-W53 - 2022-W52 eu europe european union", "World covid-19 vaccination dataset"]

# NLP method applied for finding similarity of words between the user input and the items contained within their Dictionaries
def user_selection(user_input,choices):
    answer1_nlp = nlp(user_input.upper())
    max_similarity = 0
    max_similarity_index = 0
    for i, d in enumerate(choices):
        d_nlp = nlp(d.upper())

        if answer1_nlp.similarity(d_nlp) > max_similarity:
            max_similarity = answer1_nlp.similarity(d_nlp)
            max_similarity_index = i
    if max_similarity > 0:
        return max_similarity_index
    else:
        return -1


# chatbot method which provides the user with questions and finds the similarity between the words being
# provided by the user and then applies the provided methods to display the appropriate response
def chatbot():
    print("Colin: Hi this is colin your covid-19 helper :-), what dataset will you be working with today?")

    # Find the most similar using the user_selection() method
    answer1 = input()
    dataset_choice = user_selection(answer1, datasets)
    if dataset_choice != -1:
        print("Colin: dataset found, " + dataset_locations[dataset_choice] + ". Do you wish to use this dataset?")
        answer2 = input()

        # The data_summary method() is used to provide the user with a summary of the data being provided by the Dataset
        # they are interested in, allowing the user to make an informed choice on the labels that would be most
        # appropriate for analysis.
        # note: this also identifies if the label contains (NaN) or (NULL) values, and should be dropped from ML process
        if answer2 == "yes":
            df = Data_summary.data_summary(dataset_locations[dataset_choice])
            print("Colin: Very good which task would you like to be performed on this Dataset (Machine Learning,visualisation plot) ?")
            task = input()
            task_choice = user_selection(task,tasks_performed)
            print(task_choice)
            if tasks_performed[task_choice] == "Scatter plot":
                visualisation_task(df,task,tasks_performed[task_choice],nlp)

            print("Colin: What Machine Learning model would you like to be performed on this Dataset?")
            answer3 = input()
            mlIndex = user_selection(answer3, ml_models)
            print("Colin: found Machine Learning Model " + ml_location[mlIndex] + ", Do you want to use this model?")
            if answer3 == "yes":
                print("Colin: Which Target variable are you interested in for covid 19 Dataset")
                answer4 = input()
                
chatbot()















