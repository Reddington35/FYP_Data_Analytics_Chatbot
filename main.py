import pandas as pd,spacy
from sklearn.ensemble import RandomForestClassifier

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

# testing for nlp
sentence = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
test_doc = nlp(sentence)
print([token.text for token in test_doc])

training_doc = "COVID-19_HSE_Weekly_Vaccination_Figures.csv"

training_features = ["X","Y","ExtractDate","Week","TotalweeklyVaccines","Male","Female","NA","Moderna","Pfizer","Janssen",
                     "AstraZeneca","Partial_Age0to9","Partial_Age10to19","Partial_Age20to29","Partial_Age30to39","Partial_Age40to49",
                     "Partial_Age50to59","Partial_Age60to69","Partial_Age70to79","Partial_Age80_","Partial_NA","ParCum_Age0to9",
                     "ParCum_Age10to19","ParCum_Age20to29","ParCum_Age30to39","ParCum_Age40to49","ParCum_Age50to59",
                     "ParCum_Age60to69","ParCum_Age70to79","ParCum_80_","ParCum_NA","ParPer_Age0to9","ParPer_Age10to19",
                     "ParPer_Age20to29","ParPer_Age30to39","ParPer_Age40to49","ParPer_Age50to59","ParPer_Age60to69",
                     "ParPer_Age70to79","ParPer_80_","ParPer_NA","Fully_Age0to9","Fully_Age10to19","Fully_Age20to29",
                     "Fully_Age30to39","Fully_Age40to49","Fully_Age50to59","Fully_Age60to69","Fully_Age70to79",
                     "Fully_Age80_","Fully_NA","FullyCum_Age0to9","FullyCum_Age10to19","FullyCum_Age20to29",
                     "FullyCum_Age30to39","FullyCum_Age40to49","FullyCum_Age50to59","FullyCum_Age60to69",
                     "FullyCum_Age70to79","FullyCum_80_","FullyCum_NA","FullyPer_Age0to9","FullyPer_Age10to19",
                     "FullyPer_Age20to29","FullyPer_Age30to39","FullyPer_Age40to49","FullyPer_Age50to59",
                     "FullyPer_Age60to69","FullyPer_Age70to79","FullyPer_80_","FullyPer_NA","ObjectId"]

dependent_feature = ["Week"]

read_training_data = pd.read_csv(training_doc)
print(read_training_data)

x_training = read_training_data.loc[:,training_features]
print(x_training)

y_training = read_training_data.loc[:,dependent_feature]
print(y_training)

hyper_peram_criterion = "entropy"
hyper_peram_max_depth = 20

model = RandomForestClassifier(criterion = hyper_peram_criterion,max_depth = hyper_peram_max_depth)







