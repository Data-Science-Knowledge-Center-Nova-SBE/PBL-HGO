import mysql.connector
import pandas as pd

creds = 

#Connection to the database
host = creds[2]
user = creds[0]
password = creds[1]
database = creds[3]
port = creds[4]
mydb = mysql.connector.connect(host=host, user=user, database=database, port=port, password=password, auth_plugin='mysql_native_password')
mycursor = mydb.cursor()

#Safecheck to guarantee that the connection worked
mycursor.execute('SHOW TABLES;')
print(f"Tables: {mycursor.fetchall()}")
print(mydb.connection_id) #it'll give connection_id,if got connected

alertP1 = pd.read_sql("""SELECT * FROM ConsultaUrgencia_doentespedidosconsultaNeurologia2012""",mydb)
SClinico = pd.read_sql("""SELECT * FROM consultaneurologia201216anon_true""",mydb)


text = SClinico['Texto'] 
# Create an empty list to store the text
text_list = []

# Loop through the 'text' column
for text in text.str.lower(): # Transform every word to lower case
    text_list.append(text)

### FUNCTIONS ###
def lem_pt(text):
    import spacy
    import pt_core_news_md
    nlp = pt_core_news_md.load()
    
    #Lemmatization Output
    
    document = nlp(text_list)
    lemmatized_spacy_output = " ".join([token.lemma_ for token in document])
    print(lemmatized_spacy_output)


### TEST ###
print(lem_pt(text_list))

