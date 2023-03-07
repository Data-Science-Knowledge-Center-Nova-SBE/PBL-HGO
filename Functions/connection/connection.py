import mysql.connector
import pandas as pd
#Connection to the database

def connection(creds_path):

   """
   File must follow this structure:
   "username"
   "password"
   "host"
   "database"
   port
   """
   creds = pd.read_csv(creds_path, sep=",", header=None, names=["value"])
   host = creds['value'][1]
   user = creds['value'][2]
   password = creds['value'][3]
   database = creds['value'][4]
   port = creds['value'][5]
   mydb = mysql.connector.connect(host=host, user=user, database=database, port=port, password=password, auth_plugin='mysql_native_password')
   mycursor = mydb.cursor()
   #Safecheck to guarantee that the connection worked
   mycursor.execute('SHOW TABLES;')
   print(f"Tables: {mycursor.fetchall()}")
   print(mydb.connection_id) #it'll give connection_id,if got connected
   alertP1 = pd.read_sql("""SELECT * FROM consultaneurologia201216anon_true""",mydb)
   return(alertP1)