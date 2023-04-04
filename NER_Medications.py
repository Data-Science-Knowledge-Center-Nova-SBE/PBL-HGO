import mysql.connector
import pandas as pd
from Functions.AlertP1.data_cleaning import *
from Functions.NLP.alertp1_nlp import *

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

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

print("Data Created")

#Data Cleaning
lower_text(alertP1,"Texto","clean_text")
date_format_alertP1(alertP1)
replace_blank(alertP1)
alertP1=alertP1.sort_values('DATA_RETORNO')
alertP1 = alertP1.dropna(subset=['Texto'])

print("Data Cleaned")

### NLP ###

#Medications

medications = pd.read_excel('Data/drugs_data_big2.xlsx')

from Levenshtein import distance

# Define a function to check if a word is a close match to any medication name
def is_medication(word, medications, max_distance=2):
    for medication in medications:
        if distance(word, medication) <= max_distance:
            return True
    return False

# compute the number of rows to select
n = int(len(alertP1) * 0.8)

# select the first 80% of rows using slicing
train = alertP1.iloc[:n]
#select the last 20%
test = alertP1.iloc[n:]


# Create a binary label for each row of text indicating whether it mentions a medication
train['label'] = train['Texto'].apply(lambda x: any(is_medication(word, medications) for word in x.split()))

medications = pd.read_excel('Data/drugs_data_big2.xlsx')

# Create a binary label for each row of text indicating whether it mentions a level 1 medication
train['level_1'] = train['Texto'].apply(lambda x: any(is_medication(word, medications.loc[medications['level'] == 1, 'name']) for word in x.split()))

# Create a binary label for each row of text indicating whether it mentions a level 2 medication
train['level_2'] = train['Texto'].apply(lambda x: any(is_medication(word, medications.loc[medications['level'] == 2, 'name']) for word in x.split()))

# Create a binary label for each row of text indicating whether it mentions a level 3 medication
train['level_3'] = train['Texto'].apply(lambda x: any(is_medication(word, medications.loc[medications['level'] == 3, 'name']) for word in x.split()))


# Create three pipelines to vectorize the text and train a classifier for each label
pipelines = {
    'level_1': Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ]),
    'level_2': Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ]),
    'level_3': Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
}

# Train the models on the training data
for level, pipeline in pipelines.items():
    pipeline.fit(train['Texto'], train[level])

# Make predictions on the data using the trained models
test['level_1_pred'] = pipelines['level_1'].predict(test['Texto'])
test['level_2_pred'] = pipelines['level_2'].predict(test['Texto'])
test['level_3_pred'] = pipelines['level_3'].predict(test['Texto'])



print("Medications done")

#Save the new, clean and ready to slay CSV
# load_data("data_with_meds_aas80.csv",alertP1)

# print("Data Saved")
