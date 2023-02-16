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
def spacy_lemmatizer(text):
    import spacy
    import pt_core_news_md
    nlp = pt_core_news_md.load()

    doclist = list(nlp.pipe(text))

    docs=[]
    for i, doc in enumerate(doclist):
        docs.append(' '.join([listitem.lemma_ for listitem in doc]))
        
    return docs
def remove_stopwords(text):
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Get the Portuguese stop words
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(['.', ',','(',')',':','-','?','+','/',';','drª','``','','desde','doente','consulta','alterações','se',"''",'cerca','refere','hgo','utente','vossa','s','...','ainda','c','filha','costa','dr.','pereira','ja','--','p','dr','h','n','>','q','//','..','b','++','%'])

    # Create a new list to store the filtered text
    filtered_text = []

    # Loop through the text_list and remove the stop words
    for text in text_list:
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in stop_words]
        filtered_text.append(" ".join(words))

    return filtered_text

### TEST ###
x = remove_stopwords(text_list)
spacy_lemmatizer(x)