import pandas as pd
from sklearn.ensemble import RandomForestClassifier

training_doc = "vac_rates_2021.csv"

training_features = ["STATISTIC CODE","TLIST(M1)","Month","C03898V04649","Local Electoral Area","C02076V03371","Age Group","UNIT","VALUE"]
dependent_feature = ["Statistic"]

read_training_data = pd.read_csv(training_doc)
print(read_training_data)

x_training = read_training_data.loc[:,training_features]
print(x_training)

y_training = read_training_data.loc[:,dependent_feature]
print(y_training)

hyper_peram_criterion = "gini"
hyper_peram_max_depth = 20

training_accuracy = []

model = RandomForestClassifier(criterion = hyper_peram_criterion,max_depth = hyper_peram_max_depth)





