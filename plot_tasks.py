import matplotlib.pyplot as plt
import pandas as pd
import Tasks

# World Dataset operations using different types of charts
def make_line_chart(dataset,entry_field,exit_field,target,region,location,start_date,end_date,color):
     # read in dataframe
     df = pd.read_csv(dataset['Location'],index_col=False)
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
     df = pd.read_csv(dataset,index_col=False)
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
     df = pd.read_csv(dataset,index_col=False)
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
     df = pd.read_csv(dataset, index_col=False)
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

def dynamic_line_chart(dataset,x_input,y_input,username):
     # read in dataframe
     df = pd.read_csv(dataset["Location"], index_col=False)
     print("Colin: Please enter name for the figure you are plotting")
     fig = input(username)
     print("Colin: Please enter a title for the graph")
     title = input(username)
     while x_input not in df.head(0):
          print("Colin: Please enter x feature please")
          x_input = input(username)
     while y_input not in df.head(0):
          print("Colin: Please enter y feature please")
          y_input = input(username)
     if x_input and y_input in df.head(0):
          plt.figure(fig)
          plt.plot(df[x_input], df[y_input])
          plt.title(title)
          plt.xlabel(x_input)
          plt.ylabel(y_input)
          plt.show()
     # Allow user to refine chart
     #refine = Tasks.decision_handler("Colin: Would you like to refine this graph",username)
     refine = True
     while refine:
          #print("Colin: How would you like me to update your graph " + username.replace(':','').strip() + '?')
          print("Colin: Would you like to refine this graph " + username.replace(':','').strip() + "?")
          user = input(username)
          col = "blue"
          if user.lower() in ('y', 'yes'):
               print("Colin: How would you like me to update your graph " + username.replace(':', '').strip() + '?')
               user = input(username)
               col = color_select(user)
               x_input = change_x_input(user,dataset,x_input)
               plt.figure(fig)
               plt.plot(df[x_input], df[y_input],color=col)
               plt.title(title)
               plt.xlabel(x_input)
               plt.ylabel(y_input)
               plt.show()
          if user.lower() in ('n', 'no'):
               refine = False

def dynamic_bar_chart(dataset,x_input,y_input,username):
     # read in dataframe
     df = pd.read_csv(dataset["Location"], index_col=False)
     print("Colin: Please enter name for the figure you are plotting")
     fig = input(username)
     print("Colin: Please enter a title for the graph")
     title = input(username)
     while x_input not in df.head(0):
          print("Colin: Please enter x feature please")
          x_input = input(username)
     while y_input not in df.head(0):
          print("Colin: Please enter y feature please")
          y_input = input(username)
     if x_input and y_input in df.head(0):
          plt.figure(fig)
          plt.bar(df[x_input],df[y_input])
          plt.title(title)
          plt.xlabel(x_input)
          plt.ylabel(y_input)
          plt.show()

def dynamic_scatter_chart(dataset,x_input,y_input,username):
     # read in dataframe
     df = pd.read_csv(dataset["Location"], index_col=False)
     print("Colin: Please enter name for the figure you are plotting")
     fig = input(username)
     print("Colin: Please enter a title for the graph")
     title = input(username)
     while x_input not in df.head(0):
          print("Colin: Please enter x feature please")
          x_input = input(username)
     while y_input not in df.head(0):
          print("Colin: Please enter y feature please")
          y_input = input(username)
     if x_input and y_input in df.head(0):
          plt.figure(fig)
          plt.scatter(df[x_input],df[y_input])
          plt.title(title)
          plt.xlabel(x_input)
          plt.ylabel(y_input)
          plt.show()

def dynamic_histogram(dataset,x_input,username):
     # read in dataframe
     df = pd.read_csv(dataset["Location"], index_col=False)
     print("Colin: Please enter name for the figure you are plotting")
     fig = input(username)
     print("Colin: Please enter a title for the graph")
     title = input(username)
     print("Colin: Please enter x feature please")
     x_input = input(username)
     while x_input not in df.head(0):
          print("Colin: Invalid feature, please enter a correct feature")
          x_input = input(username)
     if x_input in df.head(0):
          print("here")
          plt.figure(fig)
          plt.hist(df[x_input],bins=10)
          plt.title(title)
          plt.xlabel(x_input)
          plt.show()

def group_features(dataset):
     df = pd.read_csv(dataset['Location'], index_col=False)
     features = []
     for feature in df.head(0):
          features.append(f'{feature}')
     data = features
     return data
#group_features("datasets/covid_World.csv")

def color_select(user_input):
     color = "blue"
     color_select = ["red", "green","blue", "yellow", "purple", "orange", "cyan", "magenta", "pink", "black"]
     for i in color_select:
          if i in user_input.lower():
               color = str(i)
     return color

def change_x_input(user_input,dataset,x_input):
     x = x_input
     x_list = ["change x input","change x co-ordinate","change x","update x","update x co-ordinate",
                "change x coordinate","change x coordinate","x="]
     features = group_features(dataset)
     # check if command is looking for X
     target_x = ""
     for x1 in x_list:
          if x1 in user_input:
               # get x value at end of string
               if "=" in user_input:
                    target_x = user_input[user_input.rfind("=")+1:]
               else:
                    target_x = user_input[user_input.rfind(" ")+1:]
               if target_x in features:
                    return target_x
     return x

def commands(user_input,dataset):
     features = group_features(dataset)
     commands = ["change x","change y","filter","change figure","change title"]







