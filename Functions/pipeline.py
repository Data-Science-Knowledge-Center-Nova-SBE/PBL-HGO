

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from numpy import argsort
from Functions.AlertP1.data_cleaning import *
from Functions.AlertP1.features import *
from Functions.analysis.step_analysis import *
from Functions.AlertP1.dummy_features import *
from Functions.Models.xgboost import *
from Functions.Models.evaluation import *
from Functions.NLP.alertp1_nlp import *
from Functions.NLP.data_with_nlp import *

def pre_process(df):
    #Additional Functions

    def sort_values(df):
        return df.sort_values('DATA_RECEPCAO')

    def text_only(df):
        df=df[df['text_length']>0]
        return df

    def medication_count(df):
        df['medication_count']=df['medication_level_1']+df['medication_level_2']+df['medication_level_3']
        return df

    def medication_concentration(df):
        df['medication_concentration']=df['medication_count']/df['text_length']
        return df


    #Pipeline

    pipeline = Pipeline([
        #Data Cleaning
        ('Date Format', FunctionTransformer(date_format_alertP1)),
        ('Replace Blanks', FunctionTransformer(replace_blank)),
        ('Duplicated Entities', FunctionTransformer(entity_duplicated)),
        ('Lower Case Text ', FunctionTransformer(lowering_text)),
        ('Target Variable', FunctionTransformer(result)),
        ('Sort Values', FunctionTransformer(sort_values)),
        #Structured Features
        ('Accepted Before', FunctionTransformer(bef_accepted)),
        ('Area Classification', FunctionTransformer(class_area)),
        ('Text Length', FunctionTransformer(text_length)),
        ('Referral Steps', FunctionTransformer(referral_steps)),
        ('Speciality', FunctionTransformer(speciality)),
        ('Unit', FunctionTransformer(unit)),
        #NLP Features
        ('symptoms_column', FunctionTransformer(symptoms_column)),
        ('exams', FunctionTransformer(exams)),
        ('comorbidities', FunctionTransformer(comorbidities)),
        ('medication_column', FunctionTransformer(medication_column)),
        ('Medication Total Count', FunctionTransformer(medication_count)),
        ('Medication Concentration', FunctionTransformer(medication_concentration)),
        ('synonyms', FunctionTransformer(synonyms)),
        ('Retrieve original text', FunctionTransformer(lowering_text)),
        ('clean_text', FunctionTransformer(clean_text)),
        #Dummies
        ('Dummies', FunctionTransformer(structured_data_dummies)),
        #Keep only text rows
        ('Text Only', FunctionTransformer(text_only))
    ])

    transformed_data = pipeline.fit_transform(df)

    return transformed_data