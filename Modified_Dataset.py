import mysql.connector
import pandas as pd
from Functions.AlertP1.data_cleaning import *
from Functions.NLP.alertp1_nlp import *

    #Login Credentials
"""
File must follow this structure:
"username"
"password"
"host"
"database"
port
"""

df = pd.read_csv("credentials.txt", sep=" ", header=None, names=["Value"])

#Connection to the database
host = df["Value"][2]
user = df["Value"][0]
password = df["Value"][1]
database = df["Value"][3]
port = df["Value"][4]
mydb = mysql.connector.connect(host=host, user=user, database=database, port=port, password=password, auth_plugin='mysql_native_password')
mycursor = mydb.cursor()

#Safecheck to guarantee that the connection worked
mycursor.execute('SHOW TABLES;')
print(f"Tables: {mycursor.fetchall()}")
print(mydb.connection_id) #it'll give connection_id,if got connected

#Create a DataFrame
alertP1 = pd.read_sql("""SELECT * FROM consultaneurologia201216anon_true""",mydb)

#Data Cleaning
date_format_alertP1(alertP1)
replace_blank(alertP1)

#NLP
remove_stop_words(alertP1, "Texto", "clean_text")
spacy_lemmatizer(alertP1, "clean_text", "clean_text")
disease_class(alertP1, "clean_text")

print(alertP1.head())

#Save the new, clean and ready to slay CSV
load_data("modified_dataset.csv",alertP1)

