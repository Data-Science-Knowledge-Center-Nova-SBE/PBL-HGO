
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from numpy import argsort
from src.pre_processing.data_cleaning import *
from src.pre_processing.features_creation.features_creation_baseline import *
from src.pre_processing.features_creation.features_creation_dummies import *
from src.modelling_pipelines.modelling_functions.model_evaluation import *
from src.pre_processing.features_creation.features_creation_transformed_nlp import *
from src.modelling_pipelines.modelling_functions.logistic_regression import *


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

def final_model(alertP1):
    alertP1=pre_process(alertP1) 
    eliminate_cols=['ID_DOENTE','PROCESSO','COD_REFERENCIA','result','COD_PZ','COD_UNID_SAUDE_PROV','UNID_PROV','TIPO_UNID','COD_CTH_PRIOR','CTH_PRIOR','COD_MOTIVO_RECUSA','DES_MOTIVO_RECUSA','COD_ESPECIALIDADE','DES_ESPECIALIDADE','agrupadora','OUTRA_ENTIDADE','DATA_RECEPCAO','DATA_ENVIO','DATA_RETORNO','NUM_TAXA','ESTADO','DATA_MARCACAO','DATA_REALIZACAO','OBSERVACOES','Mês_entrada','Ano_entrada','trata data recusa','resume saída','mês_saida','ano_saida','Texto']
    X = alertP1.drop(eliminate_cols,axis=1)# Features
    y = alertP1.result # Target variable
    y_pred_train,y_pred_test,coefficients,intercept,X_train, X_test, y_train, y_test=log_regression(X,y)
    return alertP1,y_pred_train,y_pred_test,coefficients,intercept,X_train, X_test, y_train, y_test
