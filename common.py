from statistics import mode
import pandas as pd 
import numpy as np


def date_discretisation(date):
    if not pd.isna(date):
        # Do not use "-" as connector, because Pandas might read it as date or time!!
        return str(date.quarter) + "&" + str(date.dayofweek)
    else:
        return "Na"


# Read csv file as DataFrame, and drop ROW_ID column
def read_csv_no_rowid(file_path):
    df = pd.read_csv(file_path)
    df.drop(['row_id'], axis=1, inplace=True)

    return df


# check NaN value
def nan_count(df):
    print("Total columns: " + str(len(df.columns)))
    print("Total rows: " + str(len(df)))
    print("--------------")
    print(df.isnull().sum())

# calculate the time delta
def time_process(df, early_col_name, late_col_name, second_early_col_name=None):
    '''
    If first_early_col_name exist, then use late_col - first_early_col_name, 
        else, use then use late_col - second_early_col_name, else set result as NaN
    The result is the time delta, save it as the late column
    '''
    # basic date exist
    if (pd.isna(df[early_col_name]) == False) & (pd.isna(df[late_col_name]) == False):
        return abs(df[late_col_name] - df[early_col_name]).total_seconds()
    # basic date is not exist, use the second basic date
    elif (pd.isna(second_early_col_name) == False) & (pd.isna(df[late_col_name]) == False):
        return abs(df[late_col_name] - df[second_early_col_name]).total_seconds()
    # current date is not exist
    else:
        return np.NaN


# Train and choose models
from sdv.lite import TabularPreset
from sdv.tabular import GaussianCopula
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from sdv.tabular import TVAE
from sdv.evaluation import evaluate 

def build_model(constraints, train_data):
    score_dict = {}

    print("Strat training ...")
    # Tabular
    print("Tabular Preset")
    tabular_model = TabularPreset(
        name='FAST_ML',
        constraints=constraints)
    tabular_model.fit(train_data)
    tabular_sample = tabular_model.sample(len(train_data))

    # GaussianCopula
    print("Gaussian Copula")
    gaussian_model = GaussianCopula(
        constraints=constraints)
    gaussian_model.fit(train_data)
    gaussian_sample = gaussian_model.sample(len(train_data))

    # CTGAN
    print("CTGAN")
    ctgan_model = CTGAN(
        constraints=constraints, 
        cuda=True)
    ctgan_model.fit(train_data)
    ctgan_sample = ctgan_model.sample(len(train_data))

    # CopulaGAN
    print("CopulaGAN")
    copulagan_model = CopulaGAN(
        constraints=constraints, 
        cuda=True)
    copulagan_model.fit(train_data)
    copulagan_sample = copulagan_model.sample(len(train_data))

    # TVAE
    print("TVAE")
    tvae_model = TVAE(
        constraints=constraints, 
        cuda=True)
    tvae_model.fit(train_data)
    tvae_sample = tvae_model.sample(len(train_data))
    print("Training finished!")

    print("Strat evaluating ...")
    tabular_score = evaluate(tabular_sample, train_data)
    gaussian_score = evaluate(gaussian_sample, train_data)
    ctgan_score = evaluate(ctgan_sample, train_data)
    copulagan_score = evaluate(copulagan_sample, train_data)
    tvae_score = evaluate(tvae_sample, train_data)
    print("Evaluating finished!")

    score_dict['tabular'] = (tabular_model, tabular_score)
    score_dict['gaussian_copula'] = (gaussian_model, gaussian_score)
    score_dict['ctgan'] = (ctgan_model, ctgan_score)
    score_dict['copulagan'] = (copulagan_model, copulagan_score)
    score_dict['tvae'] = (tvae_model, tvae_score)

    best_model = sorted(score_dict.items(), key=lambda item: item[1][1]).pop()
    print("The best_model is " + best_model[0] + ", evaluation score is " + str(best_model[1][1]))

    return best_model[1][0]


# Save and Load model
import cloudpickle

def save_model(model, date_save_path):
    with open(date_save_path, 'wb') as f:
        cloudpickle.dump(model, f)


def load_model(date_load_path):
    with open(date_load_path, 'rb') as f:
        model = cloudpickle.load(f)
    return model
