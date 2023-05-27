
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from numpy import argsort
from Functions.AlertP1.data_cleaning import *
from Functions.AlertP1.features import *
from Functions.analysis.step_analysis import *
from Functions.AlertP1.dummy_features import *
from Functions.Models.evaluation import *
from Functions.NLP.alertp1_nlp import *
from Functions.NLP.data_with_nlp import *
from Functions.BERT.bert import *
from Functions.Chi_Squared.chi_squared import *

def pre_process(df):
    #Additional Functions

    def sort_values(df):
        df = df.sort_values('DATA_RECEPCAO')
        return df

    def text_only(df):
        df=df[df['text_length']>0]
        return df

    def medication_count(df):
        df['medication_count']=df['medication_level_1']+df['medication_level_2']+df['medication_level_3']
        return df
    def medication_concentration(df):
        df['medication_concentration']=df['medication_count']/df['text_length']
        return df
    def medication1_concentration(df):
        df['medication1_concentration']=df['medication_level_1']/df['text_length']
        return df
    
    def medication2_concentration(df):
        df['medication2_concentration']=df['medication_level_2']/df['text_length']
        return df
    
    def medication3_concentration(df):
        df['medication3_concentration']=df['medication_level_3']/df['text_length']
        return df
    
    
    def symptoms_count(df):
        df['symptoms_count']=df['symptom_0']+df['symptom_1']
        return df

    def symptoms_concentration(df):
        df['symptoms_concentration']=df['symptoms_count']/df['text_length']
        return df
    def symptoms1_concentration(df):
        df['symptoms1_concentration']=df['symptom_1']/df['text_length']
        return df
    def symptoms0_concentration(df):
        df['symptoms0_concentration']=df['symptom_0']/df['text_length']
        return df

    def exams_concentration(df):
        df['exams_concentration']=df['exam_identified']/df['text_length']
        return df
    def comorbidities_concentration(df):
        df['comorbidities_concentration']=df['comorbidity_identified']/df['text_length']
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
        ('Sort Values', FunctionTransformer(sort_values)),
        # text cleaning
        ('Lower Case Text ', FunctionTransformer(lowering_text)),
        
        #NLP meds symptoms...
        ('symptoms_column', FunctionTransformer(symptoms_column)),
        ('exams', FunctionTransformer(exams)),
        ('comorbidities', FunctionTransformer(comorbidities)),
        ('medication_column', FunctionTransformer(medication_column)),
        ('Medication Total Count', FunctionTransformer(medication_count)),
        ('Medication Concentration', FunctionTransformer(medication_concentration)),
        ('Medication1 Concentration', FunctionTransformer(medication1_concentration)),
        ('Medication2 Concentration', FunctionTransformer(medication2_concentration)),
        ('Medication3 Concentration', FunctionTransformer(medication3_concentration)),
        ('Symptoms Total Count', FunctionTransformer(symptoms_count)),
        ('Symptoms Concentration', FunctionTransformer(symptoms_concentration)),
        ('Symptoms0 Concentration', FunctionTransformer(symptoms0_concentration)),
        ('Symptoms1 Concentration', FunctionTransformer(symptoms1_concentration)),
        ('exams Concentration', FunctionTransformer(exams_concentration)),
        ('comorbidities Concentration', FunctionTransformer(comorbidities_concentration)),
        # Synonyms
        ('synonyms', FunctionTransformer(synonyms)),
        ('clean_text', FunctionTransformer(clean_text)),
        #LDA
        ('LDA', FunctionTransformer(LDA)),
        #chi_squared
        ('Chi_squared', FunctionTransformer(chi_squared)),
        #bert
        ('bert', FunctionTransformer(bert)),
        #word2vec
        ('word2vec', FunctionTransformer(w2v)),

        
        
    ])

    transformed_data = pipeline.fit_transform(df)

    return transformed_data