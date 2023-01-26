import matplotlib.pyplot as plt
import pandas as pd

def make_line_chart(dataset,entry_field,exit_field,target,region,location,start_date,end_date,color):

     df = pd.read_csv(dataset, sep=',')
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     print("result is ",df)
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (df['date'] < end_date)]
     plt.plot(df[target],color=color)
     plt.title(target, fontsize=14)
     plt.xlabel("From "+ start_date + " to " + end_date, fontsize=14)
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     plt.grid(True)
     plt.show()

make_line_chart("datasets/covid_world.csv","continent","location","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")


