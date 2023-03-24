import pandas as pd
import spacy as sp
from spacy.matcher import PhraseMatcher
import Tasks
import plot_tasks

# nlp model loaded using GPU for training the model
sp.prefer_gpu()
nlp = sp.load("en_core_web_lg")

# phrase_match function used match user input to pattern in interpretation.json
def phrase_match(user_input, dataset,username):
    matcher = PhraseMatcher(nlp.vocab)
    while True:
        for key in dataset:
            sList = []
            for sText in dataset[key]["Patterns"]:
                sList.append(nlp(str(sText).lower()))
            # matches term to list item
            matcher.add(key, sList)
        doc = nlp(user_input.lower())
        matches = matcher(doc)
        matched = [nlp.vocab.strings[s[0]] for s in matches]

        if len(matched) > 0:
            break;
        print("colin: Sorry I did not quiet catch that, could you rephrase please " + username.replace(':','').strip())
        user_input = input(username)
        print("Colin: Thank you for clearing that up")
    return dataset[matched[0]]

def label_match(user_input, dataset,username):
    matcher = PhraseMatcher(nlp.vocab)
    while True:
        for label in dataset:
            matcher.add(label, [nlp(label)])

        doc = nlp(user_input)
        matches = matcher(doc)
        matched = [nlp.vocab.strings[s[0]] for s in matches]
        if len(matched) > 0:
            break;
        print("Colin: Cannot find feature could you please type exact spelling for given feature")
        user_input = input(username)
        print("Colin: Thank you for clearing that up")
    return matched

# the geopolitical_term_check method is used to identify whether a dataset is a GPE or LOC, then if the condition is met
# it will add it to a list for use with other methods
def geopolitical_term_check(text):
    doc = nlp(text)
    gpe_list = []
    for entity in doc.ents:
        #print(entity.label_)
        if entity.label_ == "GPE" or entity.label_ == "LOC":
            gpe_list.append(entity.text)
            #print(entity.text)
    return gpe_list
#geopolitical_term_check("country is Ireland")

# the date_check method is similar to the geopolitical_term_check method only checks for dates
def date_check(text):
    doc = nlp(text)
    date_time = []
    for entity in doc.ents:
        if entity.label_ == "DATE":
            date_time.append(entity.text)
        print(date_time)
    return date_time

#print(geopolitical_term_check("scatter plot"))

# methods below used in testing, to see if a more dynamic method can be achieved using the graphing methods
# presented in sample_chart file
def region_check(user_input, dataset):
    df = pd.read_csv(dataset['Location'],index_col=False)
    if geopolitical_term_check(user_input) in df['continent'].values:
        print("In continent column")
        cont_peram = 'continent'
        cont_name = user_input
        return cont_peram,  cont_name

    elif geopolitical_term_check(user_input) in df['location'].values:
        print("In countries column")
        country_peram = 'location'
        country_name = user_input
        return country_peram, country_name
    else:
        print("Please use exact case and spelling for labels for example: Europe")

def set_start_date(user_input, dataset):
    df = pd.read_csv(dataset['Location'],index_col=False)
    if date_check(user_input) in df['date'].values:
        print("In start date column")
        date_start = user_input
        return  date_start
    else:
        print("Colin: Please use date format year-month-day, for example: 2020-02-29")

def set_end_date(user_input, dataset):
    df = pd.read_csv(dataset['Location'], index_col=False)
    if date_check(user_input) in df['date'].values:
        print("In end date column")
        date_end = user_input
        return date_end
    else:
        print("Colin: Please use date format year-month-day, for example: 2020-02-29")

def world_dataset_region(dataset, username):
    if dataset['Location'] == "datasets/covid_World.csv":
        print("\nColin: which Continent are you interested in plotting?")
        continent = input(username)
        continent_peram, continent_name = region_check(continent, dataset)
        print("\nColin: which Country are you interested in plotting?")
        country = input(username)
        country_peram, country_name = region_check(country, dataset)
        print("Colin: Please enter the start date for the time period you are interested in plotting "
              "Please use the format year-month-day, for example: 2020-02-29")
        start_date = input(username)
        start_date_plot = set_start_date(start_date, dataset)
        print("Colin: Please enter the end date for the time period you are interested in plotting "
              "Please use the format year-month-day, for example: 2020-02-29")
        end_date = input(username)
        end_date_plot = set_end_date(end_date, dataset)
        print("Colin: Which Target variable are you interested in plotting?")
        target = input(username)
        target_chosen = Tasks.find_target(target, dataset)

        decide =  Tasks.decision_handler("Colin: Would you like to plot the chart with the following information,\n"
              "continent: " + continent + ",country: " + country + ",start-date: " + start_date + ",end-date " + end_date
              + ",Target: " + target,username)

        if decide:
            plot_tasks.make_line_chart(dataset,continent_peram,country_peram,target_chosen,continent_name,country_name,start_date_plot,
                            end_date_plot,"purple")
        else:
            print("invalid chart")
        return username

