import matplotlib.pyplot as plt
import spacy as sp
import pandas as pd
import numpy as np

def visualisation_task(dataset,user_input,plot_type,nlp):
    # remove the name of the chart from the command
    user_input.replace(plot_type,"")

    # use split to separate out remaining arguments
    perams = user_input.split(" ")
    print(perams)
    if len(perams) > 1:

        # find the columns for (x,y) in the dataset
        cols = dataset.columns.values
        print(user_input)
        column_search = []

        for p in perams:
            answer1_nlp = nlp(p.upper())
            max_similarity = 0
            max_similarity_d = ""
            for i, d in enumerate(cols):
                d_nlp = nlp(d.upper())

                if answer1_nlp.similarity(d_nlp) > max_similarity:
                    max_similarity = answer1_nlp.similarity(d_nlp)
                    max_similarity_d = d

            column_search.append([max_similarity_d,max_similarity])
        print(column_search)
        # find results with max similarity
        max_col_score = 0
        max_col_index = 0

        for i,c in enumerate(column_search):
            if c[1] > max_col_score:
                max_col_score = c[1]
                max_col_index = i
        max_cols = [max_col_index]
        # find the next largest max column
        max_col_score_sec = 0
        max_col_index_sec = 0

        for i, c in enumerate(column_search):
            if c[1] > max_col_score_sec and i != max_col_index:
                max_col_score_sec = c[1]
                max_col_index_sec = i
        max_cols.append(max_col_index_sec)
        max_cols.sort()
        x = column_search[max_cols[0]][0]
        y = column_search[max_cols[1]][0]
        print(x,y)

        plt.scatter(dataset[x],dataset[y])
        plt.show()





