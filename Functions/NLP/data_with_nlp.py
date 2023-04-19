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
def symptoms_column2(alertP1):
   categorize_symptoms_simple(alertP1,"clean_text", "Data/symptoms_data_big.xlsx", 80)
   add_textcount_columns(alertP1,"clean_text","symptom_identified") 
   return(alertP1)
def exams(alertP1):
   categorize_exams(alertP1,"clean_text", "Data/exams_data.xlsx", 95)
   add_textcount_columns(alertP1,"clean_text","exam_identified") 
   return(alertP1)
def comorbidities(alertP1):
   categorize_comorbidities(alertP1,"clean_text", "Data/comorbidities_data.xlsx", 85)
   add_textcount_columns(alertP1,"clean_text","comorbidity_identified") 
   return(alertP1)
def synonyms(alertP1):
   check_synonyms("Data/synonyms_dict.xlsx", alertP1, "clean_text", 80, process_all=True)
   return(alertP1)

    