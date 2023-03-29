from Functions.NLP.alertp1_nlp import *

### NLP ###

#Medications
def medication_column(alertP1):
   alertP1["clean_text"] = alertP1["Texto"].copy()
   categorize_medication(alertP1,"clean_text", "Data/drugs_data.xlsx", 75)
   add_textcount_columns(alertP1,"clean_text","medication_level_1")
   add_textcount_columns(alertP1,"clean_text","medication_level_2")
   add_textcount_columns(alertP1,"clean_text","medication_level_3")
   remove_stop_words(alertP1, "clean_text", "clean_text")
   spacy_lemmatizer(alertP1, "clean_text", "clean_text")
   return(alertP1)

