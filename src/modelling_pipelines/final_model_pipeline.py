
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from numpy import argsort
from src.pre_processing.data_cleaning import *
from src.pre_processing.features_creation.features_creation_baseline import *
from src.pre_processing.features_creation.features_creation_dummies import *
from src.modelling_pipelines.modelling_functions.model_evaluation import *
from src.pre_processing.features_creation.features_creation_nlp import *
from src.pre_processing.features_creation.features_creation_transformed_nlp import *
#from src.pre_processing.features_creation.features_creation_bert import *
from src.pre_processing.features_creation.features_creation_chisquared import *
from src.modelling_pipelines.modelling_functions.xgboost import *


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
        
        # #NLP meds symptoms...
        # ('symptoms_column', FunctionTransformer(symptoms_column)),
        # ('exams', FunctionTransformer(exams)),
        # ('comorbidities', FunctionTransformer(comorbidities)),
        # ('medication_column', FunctionTransformer(medication_column)),
        # ('Medication Total Count', FunctionTransformer(medication_count)),
        # ('Medication Concentration', FunctionTransformer(medication_concentration)),
        # ('Medication1 Concentration', FunctionTransformer(medication1_concentration)),
        # ('Medication2 Concentration', FunctionTransformer(medication2_concentration)),
        # ('Medication3 Concentration', FunctionTransformer(medication3_concentration)),
        # ('Symptoms Total Count', FunctionTransformer(symptoms_count)),
        # ('Symptoms Concentration', FunctionTransformer(symptoms_concentration)),
        # ('Symptoms0 Concentration', FunctionTransformer(symptoms0_concentration)),
        # ('Symptoms1 Concentration', FunctionTransformer(symptoms1_concentration)),
        # ('exams Concentration', FunctionTransformer(exams_concentration)),
        # ('comorbidities Concentration', FunctionTransformer(comorbidities_concentration)),
        # ('synonyms', FunctionTransformer(synonyms)),

        # #lemmatizating and removing stop words
        # ('clean_text', FunctionTransformer(clean_text)),
        # #chi_squared
        # ('Chi_squared', FunctionTransformer(chi_squared)),

        
        
    ])

    transformed_data = pipeline.fit_transform(df)

    return transformed_data

def final_model(alertP1):
    alertP1=pre_process(alertP1)
    eliminate_cols=['ID_DOENTE','PROCESSO','COD_REFERENCIA','result','COD_PZ','COD_UNID_SAUDE_PROV','UNID_PROV','TIPO_UNID','COD_CTH_PRIOR','CTH_PRIOR','COD_MOTIVO_RECUSA','DES_MOTIVO_RECUSA','COD_ESPECIALIDADE','DES_ESPECIALIDADE','agrupadora','OUTRA_ENTIDADE','DATA_RECEPCAO','DATA_ENVIO','DATA_RETORNO','NUM_TAXA','ESTADO','DATA_MARCACAO','DATA_REALIZACAO','OBSERVACOES','Mês_entrada','Ano_entrada','trata data recusa','resume saída','mês_saida','ano_saida','Texto','clean_text_caveman','clean_text','chi_squared']    
    X = alertP1.drop(eliminate_cols,axis=1)# Features
    y = alertP1.result # Target variable
    y_pred_train, y_pred_test, model_score, X_train, X_test, y_train, y_test = xgb_classifier(X,y)
    return alertP1,y_pred_train, y_pred_test, model_score, X_train, X_test, y_train, y_test 