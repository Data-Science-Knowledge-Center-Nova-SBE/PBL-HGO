import mysql.connector
import pandas as pd
#Connection to the database
def connection(creds):
   host = creds['value'][0]
   user = creds['value'][1]
   password = creds['value'][2]
   database = creds['value'][3]
   port = creds['value'][4]
   mydb = mysql.connector.connect(host=host, user=user, database=database, port=port, password=password, auth_plugin='mysql_native_password')
   mycursor = mydb.cursor()
   #Safecheck to guarantee that the connection worked
   mycursor.execute('SHOW TABLES;')
   print(f"Tables: {mycursor.fetchall()}")
   print(mydb.connection_id) #it'll give connection_id,if got connected
   alertP1 = pd.read_sql("""SELECT * FROM consultaneurologia201216anon_true""",mydb)
   return(alertP1)