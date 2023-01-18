import pandas as pd
import numpy as np
def data_summary(dataset):
    df = pd.read_csv(dataset, sep=',')
    #df = pd.DataFrame(dataset)
    print("Data Summary:")
    print("The number of rows in this Dataset are " + str(len(df.index)))
    print("The number of columns in this dataset are " + str(len(df.columns)))
    print("The shape of this Dataset is " + str(df.shape))
    print("\nThe available labels for learning are: \n")

    for i in df.head(0):
        print(i + ", ",sep="", end='')
    print("\n*****************************\n")
#data_summary("datasets/covid_Ireland.csv")



