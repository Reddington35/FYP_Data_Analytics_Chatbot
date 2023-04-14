# FYP_Data_Analytics_Chatbot
Final Year Project - Data Analytics Chatbot:

The project consists of a user friendly chatbot titled Colin, that provides the user with various methods of asertaining information about the 
Covid 19 Pandemic for my University project.

Key Features:
- Main Menu loop used to guide the user through the key features using Natural Language Processing (NLP) from the SpaCy Library.
- Graphing options are Scatter Plot, Line Graph, Bar Chart and a histogram. All graphs use a dynamic graphing system which allows the user
  to provide a name to the figure, a title for the graph, change the colour, Bins (Histogram Only) X or Y features, as well as an option to filter   the dataset using the opperators (<,>,=). The option filter reset can then be used to reset the dataset to its initial state. 
- Machine Learning tasks allow the user to perform ML tasks on a finite number of datasets (4) in total, performing these tasks using the
  Scikit Learn (API) using the Classification algorithms,
  (Random Forest Classifier, Naive Bayes GaussianNB Classifier and Random Forest Regressor), this feature is also dynamic allowing the user the     option to change Hyper - Perameter values as they see fit.
- The chatbot will also provide the user options to show the available Features of each dataset, and provides a summary of the available data,
  for example Datatypes, labels, values etc.
- The user will be able to move between datasets from the main menu loop as they please.

On Leaving the application the user will be asked if they wish to provide any feedback to the developer, 
so further improvements can be made to the design.
The application will also keep track of previous relevant queries the user has provided for future use and is stored as a file according to the session date. A Help feature is provided in the package for all available features.

