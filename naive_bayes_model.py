import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

option = ["iso_code","continent","location","date,total_cases"]
data = pd.read_csv("datasets/covid_World.csv", sep=',')
data = data[["new_cases","new_cases_smoothed","total_deaths","new_deaths",
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
             "excess_mortality_cumulative_per_million"]]

predict = "cardiovasc_death_rate"

x = np.array(data.drop([predict],axis=1))
y = np.array(data[predict])

#x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.10,random_state=1)
#model = RandomForestClassifier(min_samples_split=3,n_estimators=6,max_depth=20,max_features=6)

parameters = {"min_samples_split" :[2,8]}
gaussian_classifier = GaussianNB()
gridsearch = GridSearchCV(gaussian_classifier,parameters)
gridsearch.fit(x,y)
print(sorted(gridsearch.cv_results_.keys()))
print(gridsearch.best_params_)
print(gridsearch.best_score_)

#model.fit(x_train, y_train)
#dump(model,open('model.csv','wb'))
#accuracy = model.score(x_test, y_test)

#predictions = model.predict(x_test)
#for x in range(len(predictions)):
 #   print("Predictions: " + str(predictions[x]) + "\n",
  #        "features test Data: " + str(x_test[x]) +"\n",
   #       "Target test Data: " + str(y_test[x]) + "\n")

#print(accuracy)import pandas as pd