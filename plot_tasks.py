import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# World Dataset operations using different types of charts
# Function for plotting a Line Chart
def make_line_chart(dataset,entry_field,exit_field,target,region,location,start_date,end_date,color):
     # read in dataframe
     df = pd.read_csv(dataset['Location'], sep=',')
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     #print("result is ",df)
     # converts date string to date object using the pandas library
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     # Filters dataframe by region, location, start date and end date
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (df['date'] < end_date)]
     # sets plot using target varible, color set in perameters
     plt.plot(df[target],color=color)
     # sets the title of the chart
     plt.title(target, fontsize=14)
     # labels the x values
     plt.xlabel("From "+ start_date + " to " + end_date, fontsize=14)
     # labels the y values
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     # sets to grid view
     plt.grid(True)
     # displays plot
     plt.show()

#make_line_chart("datasets/covid_world.csv","continent","location","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

# Function for plotting a Scatter Plot
def make_scatterplot(dataset,entry_field,exit_field,x,y,region,location,start_date,end_date,color):
     df = pd.read_csv(dataset, sep=',')
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     print("result is ", df)
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (
                  df['date'] < end_date)]
     plt.scatter(df[x],df[y], color=color)
     plt.title("comparing " + x + " to " + y, fontsize=14)
     plt.xlabel("From " + start_date + " to " + end_date, fontsize=14)
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     plt.grid(True)
     plt.show()

#make_scatterplot("datasets/covid_world.csv","continent","location","total_deaths","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

# Function for plotting Bar Chart
def make_barchart(dataset,entry_field,exit_field,x,y,region,location,start_date,end_date,color):
     df = pd.read_csv(dataset, sep=',')
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     print("result is ", df)
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (
             df['date'] < end_date)]
     plt.bar(df[x], df[y], color=color)
     plt.title("comparing " + x + " to " + y, fontsize=14)
     plt.xlabel("From " + start_date + " to " + end_date, fontsize=14)
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     plt.grid(True)
     plt.show()

#make_barchart("datasets/covid_world.csv","continent","location","date","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

# Function for plotting Bar Chart
def make_histogram(dataset,entry_field,exit_field,x,region,location,start_date,end_date,color):
     df = pd.read_csv(dataset, sep=',')
     df = df[(df[entry_field] == region) & (df[exit_field] == location)]
     print("result is ", df)
     df['date'] = pd.to_datetime(df['date'])
     df = df.set_index(df['date'])
     df = df[(df[entry_field] == region) & (df[exit_field] == location) & (df['date'] > start_date) & (
             df['date'] < end_date)]
     df.hist(x, figsize=[12,12],bins=12,color=color)
     plt.title("Title:" + x, fontsize=14)
     plt.xlabel("From " + start_date + " to " + end_date, fontsize=14)
     plt.ylabel("Region " + region + " - " + location, fontsize=14)
     plt.grid(True)
     plt.show()
#make_histogram("datasets/covid_world.csv","continent","location","new_deaths","Europe","Ireland","2020-02-29","2023-01-15","purple")

