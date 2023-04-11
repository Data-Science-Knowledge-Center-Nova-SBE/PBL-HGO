from Functions.NLP.alertp1_nlp import *

### NLP ###
def lowering_text(alertP1):
    lower_text(alertP1,"Texto","clean_text")
    return(alertP1)
def clean_text(alertP1):
   remove_stop_words(alertP1, "clean_text", "clean_text")
   spacy_lemmatizer(alertP1, "clean_text", "clean_text")
   return(alertP1)
#Medications
def medication_column(alertP1):
   categorize_medication(alertP1,"clean_text", "Data/drugs_data_big2.xlsx", 80)
   add_textcount_columns(alertP1,"clean_text","medication_level_1")
   add_textcount_columns(alertP1,"clean_text","medication_level_2")
   add_textcount_columns(alertP1,"clean_text","medication_level_3")   
   return(alertP1)
#Symptoms
def symptoms_column(alertP1):
   categorize_symptoms(alertP1,"clean_text", "Data/symptoms_data.xlsx", 80)
   add_textcount_columns(alertP1,"clean_text","symptom_1")
   add_textcount_columns(alertP1,"clean_text","symptom_0")  
   return(alertP1)
