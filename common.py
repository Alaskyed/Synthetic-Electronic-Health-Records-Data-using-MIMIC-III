import pandas as pd 


def date_discretisation(date):
    if not pd.isna(date):
        # Do not use "-" as connector, because Pandas might read it as date or time!!
        return str(date.quarter) + "&" + str(date.dayofweek)
    else:
        return "Na"


# read csv file as dataframe, and drop ROW_ID column
def read_csv_no_rowid(file_path):
    df = pd.read_csv(file_path)
    df.drop(["Unnamed: 0", 'row_id'], axis=1, inplace=True)

    return df


# check NaN value
def nan_count(df):
    print("Total columns: " + str(len(df.columns)))
    print("Total rows: " + str(len(df)))
    print("--------------")
    print(df.isnull().sum())

