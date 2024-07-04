import mysql.connector

# Paramètres de connexion à la base de données
host = 'localhost'
database = 'cefe'
user = 'root'
password = ''

# Fonction pour obtenir la connexion à la base de données
def get_BD():
    try:
        conn = mysql.connector.connect(host=host, database=database, user=user, password=password)
        return conn
    except mysql.connector.Error as e:
        return None
