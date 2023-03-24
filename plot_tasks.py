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
          try:
               # read in dataframe
               df = pd.read_csv(dataset["Location"], index_col=False)
               print("Colin: Please enter name for the figure you are plotting")
               fig = input(username)
               print("Colin: Please enter a title for the graph")
               title = input(username)
               while x_input not in df.head(0):
                    Tasks.feature_selection(dataset)
                    print("Colin: Please enter x feature please")
                    x_input = input(username)
               while y_input not in df.head(0):
                    Tasks.feature_selection(dataset)
                    print("Colin: Please enter y feature please")
                    y_input = input(username)
               if x_input and y_input in df.head(0):
                    #labels = df[x_input]
                    plt.figure(fig)
                    plt.plot(df[x_input], df[y_input])
                    plt.title(title)
                    plt.xlabel(x_input)
                    plt.ylabel(y_input)
                    # plt.tick_params(axis='both', labelsize=7)
                    # plt.xticks(df[x_input],labels,rotation=90)
                    # plt.margins(0.2)
                    plt.show()
               # Allow user to refine chart
               refine = True
               while refine:
                    Tasks.feature_selection(dataset)
                    print("Colin: How would you like me to Refine this graph " + username.replace(':','').strip() + "?\n"
                          "Colin: Type 'help' if you require any assistance with this feature\n")
                    user_command = input(username)
                    if "help" in user_command.lower():
                         help_screen(dataset)

                    # filter command
                    #filter_dataset(dataset, user_command, username)
                    if user_command.lower().strip() not in ('quit'):
                         commands_list = user_command.split(",")
                         col = "blue"
                         for c in commands_list:
                              if len(c.lower().strip()) > 3:
                                   col = color_select(c)
                                   is_refined = False
                                   new_x_input = change_x_input(c, dataset, x_input)
                                   if new_x_input != x_input:
                                        print("Colin: Updated X from " + x_input + " to " + new_x_input)
                                        x_input = new_x_input
                                        is_refined = True
                                   if not is_refined:
                                        new_y_input = change_y_input(c, dataset, y_input)
                                        if new_y_input != y_input:
                                             print("Colin: Updated Y from " + y_input + " to " + new_y_input)
                                             y_input = new_y_input
                                             is_refined = True
                                   if not is_refined:
                                        new_title = change_title_input(c, title)
                                        if new_title != title:
                                             print("Colin: Updated title from " + title + " to " + new_title)
                                             title = new_title
                                             is_refined = True
                                   if not is_refined:
                                        new_fig = change_figure_input(c, fig)
                                        if new_fig != fig:
                                             print("Colin: Updated figure from " + fig + " to " + new_fig)
                                             fig = new_fig
                                             is_refined = True
                         plt.figure(fig)
                         plt.plot(df[x_input], df[y_input], color=col)
                         plt.title(title)
                         plt.xlabel(x_input)
                         plt.ylabel(y_input)
                         plt.show()
                    else:
                         refine = False
          except:
               print("Colin: oops something went wrong with your graph,\n please leave a comment in the feedback section about"
                    " this issue and I will try to fix it as soon as I can")

def dynamic_bar_chart(dataset,x_input,y_input,username):
     try:
          # read in dataframe
          df = pd.read_csv(dataset["Location"], index_col=False)
          print("Colin: Please enter name for the figure you are plotting")
          fig = input(username)
          print("Colin: Please enter a title for the graph")
          title = input(username)
          while x_input not in df.head(0):
               Tasks.feature_selection(dataset)
               print("Colin: Please enter x feature please")
               x_input = input(username)
          while y_input not in df.head(0):
               Tasks.feature_selection(dataset)
               print("Colin: Please enter y feature please")
               y_input = input(username)
          if x_input and y_input in df.head(0):
               plt.figure(fig)
               plt.bar(df[x_input],df[y_input])
               plt.title(title)
               plt.xlabel(x_input)
               plt.ylabel(y_input)
               plt.show()
          # Allow user to refine chart
          refine = True
          while refine:
               Tasks.feature_selection(dataset)
               print("Colin: How would you like me to Refine this graph " + username.replace(':', '').strip() + "?\n"
                     "Colin: Type 'help' if you require any assistance with this feature")
               user_command = input(username)
               if "help" in user_command.lower():
                    help_screen(dataset)

               if user_command.lower().strip() not in ('quit'):
                    commands_list = user_command.split(",")
                    col = "blue"
                    for c in commands_list:
                         if len(c.lower().strip()) > 3:
                              col = color_select(c)
                              is_refined = False
                              new_x_input = change_x_input(c, dataset, x_input)
                              if new_x_input != x_input:
                                   print("Colin: Updated X from " + x_input + " to " + new_x_input)
                                   x_input = new_x_input
                                   is_refined = True
                              if not is_refined:
                                   new_y_input = change_y_input(c, dataset, y_input)
                                   if new_y_input != y_input:
                                        print("Colin: Updated Y from " + y_input + " to " + new_y_input)
                                        y_input = new_y_input
                                        is_refined = True
                              if not is_refined:
                                   new_title = change_title_input(c, title)
                                   if new_title != title:
                                        print("Colin: Updated title from " + title + " to " + new_title)
                                        title = new_title
                                        is_refined = True
                              if not is_refined:
                                   new_fig = change_figure_input(c, fig)
                                   if new_fig != fig:
                                        print("Colin: Updated figure from " + fig + " to " + new_fig)
                                        fig = new_fig
                                        is_refined = True
                    plt.figure(fig)
                    plt.bar(df[x_input], df[y_input], color=col)
                    plt.title(title)
                    plt.xlabel(x_input)
                    plt.ylabel(y_input)
                    plt.show()
               else:
                    refine = False
     except:
          print("Colin: oops something went wrong with your graph,\n please leave a comment in the feedback section about"
                " this issue and I will try to fix it as soon as I can")

def dynamic_scatter_chart(dataset,x_input,y_input,username):
     try:
          # read in dataframe
          df = pd.read_csv(dataset["Location"], index_col=False)
          print("Colin: Please enter name for the figure you are plotting")
          fig = input(username)
          print("Colin: Please enter a title for the graph")
          title = input(username)
          while x_input not in df.head(0):
               Tasks.feature_selection(dataset)
               print("Colin: Please enter x feature please")
               x_input = input(username)
          while y_input not in df.head(0):
               Tasks.feature_selection(dataset)
               print("Colin: Please enter y feature please")
               y_input = input(username)
          if x_input and y_input in df.head(0):
               plt.figure(fig)
               plt.scatter(df[x_input],df[y_input])
               plt.title(title)
               plt.xlabel(x_input)
               plt.ylabel(y_input)
               plt.show()
          # Allow user to refine chart
          refine = True
          while refine:
               Tasks.feature_selection(dataset)
               print("Colin: How would you like me to Refine this graph " + username.replace(':', '').strip() + "?\n"
                    "Colin: Type 'help' if you require any assistance with this feature")
               user_command = input(username)

               if "help" in user_command.lower():
                    help_screen(dataset)

               if user_command.lower().strip() not in ('quit'):
                    commands_list = user_command.split(",")
                    col = "blue"
                    for c in commands_list:
                         if len(c.lower().strip()) > 3:
                              is_refined = False
                              new_x_input = change_x_input(c, dataset, x_input)
                              if new_x_input != x_input:
                                   print("Colin: Updated X from " + x_input + " to " + new_x_input)
                                   x_input = new_x_input
                                   is_refined = True
                              if not is_refined:
                                   new_y_input = change_y_input(c, dataset, y_input)
                                   if new_y_input != y_input:
                                        print("Colin: Updated Y from " + y_input + " to " + new_y_input)
                                        y_input = new_y_input
                                        is_refined = True
                              if not is_refined:
                                   new_title = change_title_input(c, title)
                                   if new_title != title:
                                        print("Colin: Updated title from " + title + " to " + new_title)
                                        title = new_title
                                        is_refined = True
                              if not is_refined:
                                   new_fig = change_figure_input(c, fig)
                                   if new_fig != fig:
                                        print("Colin: Updated figure from " + fig + " to " + new_fig)
                                        fig = new_fig
                                        is_refined = True
                    col = color_select(user_command.lower())
                    plt.figure(fig)
                    plt.scatter(df[x_input], df[y_input], color=col)
                    plt.title(title)
                    plt.xlabel(x_input)
                    plt.ylabel(y_input)
                    plt.show()
               else:
                    refine = False
     except:
          print("Colin: oops something went wrong with your graph,\n please leave a comment in the feedback section about"
                " this issue and I will try to fix it as soon as I can")

def dynamic_histogram(dataset,x_input,username):
     try:
          # read in dataframe
          df = pd.read_csv(dataset["Location"], index_col=False)
          print("Colin: Please enter name for the figure you are plotting")
          fig = input(username)
          print("Colin: Please enter a title for the graph")
          title = input(username)
          while x_input not in df.head(0):
               Tasks.feature_selection(dataset)
               print("Colin: Please enter x feature please")
               x_input = input(username)
               print("Colin: Please enter number of bins please")
               bin = input(username)
          if x_input in df.head(0):
               plt.figure(fig)
               plt.hist(df[x_input],bins=int(bin))
               plt.title(title)
               plt.xlabel(x_input)
               plt.show()
          # Allow user to refine chart
          refine = True
          while refine:
               Tasks.feature_selection(dataset)
               print("Colin: How would you like me to Refine this graph " + username.replace(':', '').strip() + "?\n"
                    "Colin: Type 'help' if you require any assistance with this feature")
               user_command = input(username)
               if "help" in user_command.lower():
                    help_screen(dataset)

               if user_command.lower().strip() not in ('quit'):
                    commands_list = user_command.split(",")
                    col = "blue"
                    for c in commands_list:
                         if len(c.lower().strip()) > 3:
                              col = color_select(user_command.lower())
                              is_refined = False
                              new_x_input = change_x_input(c, dataset, x_input)

                              if new_x_input != x_input:
                                   print("Colin: Updated X from " + x_input + " to " + new_x_input)
                                   x_input = new_x_input
                                   is_refined = True

                              if not is_refined:
                                   new_title = change_title_input(c, title)
                                   if new_title != title:
                                        print("Colin: Updated title from " + title + " to " + new_title)
                                        title = new_title
                                        is_refined = True

                              if not is_refined:
                                   new_fig = change_figure_input(c, fig)
                                   if new_fig != fig:
                                        print("Colin: Updated figure from " + fig + " to " + new_fig)
                                        fig = new_fig
                                        is_refined = True

                              if not is_refined:
                                   new_bin = change_bin_input(c, bin)
                                   if new_bin != bin:
                                        print("Colin: Updated bins from " + bin + " to " + new_bin)
                                        bin = new_bin
                                        is_refined = True
                    plt.figure(fig)
                    plt.hist(df[x_input], color=col,bins=int(bin))
                    plt.title(title)
                    plt.xlabel(x_input)
                    plt.show()
               else:
                    refine = False
     except:
          print("Colin: oops something went wrong with your graph,\n please leave a comment in the feedback section about"
                " this issue and I will try to fix it as soon as I can")

def group_features(dataset):
     df = pd.read_csv(dataset['Location'], index_col=False)
     features = []
     for feature in df.head(0):
          features.append(f'{feature}')
     data = features
     return data
#group_features("datasets/covid_World.csv")

def color_select(user_input):
     # Default color set
     color = "blue"
     color_select = ["red", "green","blue", "yellow", "purple", "orange", "cyan", "magenta", "pink", "black"]
     # loops through all colors until matched with user input
     for i in color_select:
          if i in user_input.lower():
               color = str(i)
     return color

def change_x_input(user_input,dataset,x_input):
     x = x_input
     x_list = ["change x input","change x co-ordinate","change x","update x","update x co-ordinate",
                "change x coordinate","change x coordinate","x="]
     features = group_features(dataset)
     # check if command is looking for x
     target_x = ""
     for x1 in x_list:
          if x1 in user_input:
               # get x value at end of string
               if "=" in user_input:
                    target_x = user_input[user_input.rfind("=")+1:]
               else:
                    # assume space seperates value
                    target_x = user_input[user_input.rfind(" ")+1:]
               if target_x in features:
                    return target_x
     return x


def change_y_input(user_input,dataset,y_input):
     y = y_input
     y_command =["y=","y =","change y to","update y to"]
     features = group_features(dataset)
     # check if command is looking for Y
     target_y = ""
     for y1 in y_command:
          if y1 in user_input:
               # get y value at end of string
               if "=" in user_input:
                    target_y = user_input[user_input.rfind("=")+1:]
               else:
                    target_y = user_input[user_input.rfind(" ")+1:]
               if target_y in features:
                    return target_y
     return y

def change_title_input(user_input,t_input):
     t = t_input
     titles = ["title="]
     # check if command is looking for title
     target_title = ""
     for t1 in titles:
          if t1 in user_input:
               # get title string at end of string
               if "=" in user_input:
                    target_title = user_input[user_input.rfind("=")+1:]
               else:
                    target_title = user_input[user_input.rfind(" ")+1:]
               return target_title
     return t

def change_figure_input(user_input,f_input):
     f = f_input
     figure = ["figure="]
     # check if command is looking for figure
     target_figure = ""
     for f1 in figure:
          if f1 in user_input:
               # get figure at end of string
               if "=" in user_input:
                    target_figure = user_input[user_input.rfind("=")+1:]
               else:
                    target_figure = user_input[user_input.rfind(" ")+1:]
               return target_figure
     return f

def change_bin_input(user_input,b_input):
     b = b_input
     bins = ["bins="]
     # check if command is looking for bins
     target_bins = ""
     for b1 in bins:
          if b1 in user_input:
               # get bin value at end of string
               if "=" in user_input:
                    target_bins = user_input[user_input.rfind("=") + 1:]
               else:
                    target_bins = user_input[user_input.rfind(" ") + 1:]
               return target_bins
     return b

def help_screen(dataset):
     #print Features, with clear instructions for each command needed for graph refinement process
     Tasks.feature_selection(dataset)
     print("Colin: You can refine the graph by placing '=' in front of the graph feature you wish to update\n"
                "for Example:\n"
                "-x=Feature Name\n-y=Feature Name\n-title=Name of Chart\n-figure=Name of Figure\n-bins=number of bins"
                " (Histogram only)"
                "-Fields are ',' separated\n"
                "For Example:  title=Title Name,figure=Name of Figure\n"
                "Colin: If your finished refining the graph please reply 'quit' as your response\n")

def filter_dataset(dataset,command,username):
     features = group_features(dataset)
     #filter_greater = ["filter by"]
     if command[0:6] == "filter":
          command = command[7:]
         # Check for opperators (>,<,=,<>)
          if ">" in command:
               column = command[0: command.find(">")].lower().strip()
               value = command[command.find(">")+1:].lower().strip()
               print(column + "by" + value)
          elif "<" in command:
               column = command[0: command.find("<")].lower().strip()
               value = command[command.find("<")+1:].lower().strip()
               print(column + "by" + value)
          elif "=" in command:
               column = command[0: command.find("=")].lower().strip()
               value = command[command.find("=")+1:].lower().strip()
               print(column,value)










