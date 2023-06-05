
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from numpy import argsort
from src.pre_processing.data_cleaning import *
from src.pre_processing.features_creation.features_creation_baseline import *
from src.pre_processing.step_analysis import *
from src.pre_processing.features_creation.features_creation_dummies import *
from src.modelling_pipelines.modelling_functions.model_evaluation import *
from src.pre_processing.features_creation.features_creation_transformed_nlp import *

def pre_process(df):
    #Additional Functions

    def sort_values(df):
        df = df.sort_values('DATA_RECEPCAO')
        return df
    def text_only(df):
        df=df[df['text_length']>0]
        return df

   

    #Pipeline

    pipeline = Pipeline([
        #Data Cleaning
        ('Date Format', FunctionTransformer(date_format_alertP1)),
        ('Replace Blanks', FunctionTransformer(replace_blank)),
        ('Duplicated Entities', FunctionTransformer(entity_duplicated)),
        ('Target Variable', FunctionTransformer(result)),
        #Structured Features
        ('Accepted Before', FunctionTransformer(bef_accepted)),
        ('Area Classification', FunctionTransformer(class_area)),
        ('Text Length', FunctionTransformer(text_length)),
        ('Referral Steps', FunctionTransformer(referral_steps)),
        ('Speciality', FunctionTransformer(speciality)),
        ('Unit', FunctionTransformer(unit)),
        #Dummies
        ('Dummies', FunctionTransformer(structured_data_dummies)),
        #Keep only text rows
        ('Text Only', FunctionTransformer(text_only)),
        #Sort Values
        ('Sort Values', FunctionTransformer(sort_values))
    ])
    transformed_data = pipeline.fit_transform(df)

    return transformed_data