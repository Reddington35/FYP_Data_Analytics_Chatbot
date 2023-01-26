import pandas as pd
import numpy as np

# The data_summary method provides the user with a clean representation of the data being provided by the Datasets
# they may be interested in analysing, this method provides the user with the number of Rows,columns,and shape of the
# desired Dataset they are working with while also providing the type of Data provided in each label
def data_summary(dataset):
    df = pd.read_csv(dataset, sep=',')
    print("Data Summary:")
    print("The number of rows in this Dataset are " + str(len(df.index)))
    print("The number of columns in this dataset are " + str(len(df.columns)))
    print("The shape of this Dataset is " + str(df.shape))
    print("\nThe available labels for learning are: \n")

    for i in df.head(0):
        print("Label: " + i + " " + "|Type: " + str(df.dtypes[i]) + " |value contained in first Row:" + str(df[i][0]) + "|")
    print("\n*****************************************************************************************************************\n")
    return df



