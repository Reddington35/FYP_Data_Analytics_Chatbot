import pandas as pd
import requests
import numpy as np
import spacy as sp
import sklearn
from sklearn.ensemble import RandomForestClassifier

import Data_summary

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
            Data_summary.data_summary(dataset_locations[dataset_choice])
            print("Colin: What Machine Learning model would you like to be  performed on this Dataset?")
            answer3 = input()
            mlIndex = user_selection(answer3, ml_models[dataset_choice])

            print("Colin: found Machine Learning Model " + ml_location[mlIndex] + ", Do you want to use this model?")
            if answer3 == "yes":
                print("Colin: Which Target variable are you interested in for covid 19 Dataset")
                answer4 = input()
                
chatbot()
















