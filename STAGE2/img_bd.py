from BD import get_BD

def ajt_comment(url, commentaire, username, activ):
    # Obtenez la connexion à la base de données
    conn = get_BD()
    if conn is None or activ != 1:
        return False
    
    try:
        # Créez un curseur pour exécuter les requêtes SQL
        cursor = conn.cursor()

        # Requête SQL pour insérer les données dans la table "images"
        sql = "INSERT INTO images (url, commentaire, username) VALUES (%s, %s, %s)"
        values = (url, commentaire, username)

        # Exécutez la requête SQL avec les valeurs fournies
        cursor.execute(sql, values)

        # Validez la transaction en enregistrant les modifications dans la base de données
        conn.commit()
        
        # Fermez le curseur et la connexion
        cursor.close()
        conn.close()

        return True  # Retournez True pour indiquer que l'insertion a réussi

    except Exception as e:
        # En cas d'erreur, annulez la transaction et affichez l'erreur
        print(f"Erreur lors de l'insertion dans la base de données : {e}")
        conn.rollback()
        return False
    

def ajt_img(url, username):
    # Obtenez la connexion à la base de données
    conn = get_BD()
    if conn is None :
        return False
    
    try:
        # Créez un curseur pour exécuter les requêtes SQL
        cursor = conn.cursor()

        # Requête SQL pour insérer les données dans la table "images"
        sql = "INSERT INTO img (img_url, user) VALUES (%s, %s)"
        values = (url, username)

        # Exécutez la requête SQL avec les valeurs fournies
        cursor.execute(sql, values)

        # Validez la transaction en enregistrant les modifications dans la base de données
        conn.commit()
        
        # Fermez le curseur et la connexion
        cursor.close()
        conn.close()

        return True  # Retournez True pour indiquer que l'insertion a réussi

    except Exception as e:
        # En cas d'erreur, annulez la transaction et affichez l'erreur
        print(f"Erreur lors de l'insertion dans la base de données : {e}")
        conn.rollback()
        return False
    
    
def ajt_label(id, label):
    # Obtenez la connexion à la base de données
    conn = get_BD()
    if conn is None :
        return False
    
    try:
        # Créez un curseur pour exécuter les requêtes SQL
        cursor = conn.cursor()

        # Requête SQL pour insérer les données dans la table "images"
        sql = "UPDATE img SET label = %s WHERE img_id = %s"
        values = (label, id)

        # Exécutez la requête SQL avec les valeurs fournies
        cursor.execute(sql, values)

        # Validez la transaction en enregistrant les modifications dans la base de données
        conn.commit()
        
        # Fermez le curseur et la connexion
        cursor.close()
        conn.close()

        return True  # Retournez True pour indiquer que l'insertion a réussi

    except Exception as e:
        # En cas d'erreur, annulez la transaction et affichez l'erreur
        print(f"Erreur lors de l'insertion dans la base de données : {e}")
        conn.rollback()
        return False


def get_random_image_from_db(n):
    # Obtenez la connexion à la base de données
    conn = get_BD()
    if conn is None:
        return False, None  # Retourner False si la connexion échoue
    
    try:
        cursor = conn.cursor()
        if n == 3:
            cursor.execute("SELECT img_id, img_url FROM img WHERE label = '' ORDER BY RAND() LIMIT 3;")
            result = cursor.fetchall()  # Récupérer les IDs et URLs de trois images
        else:
            cursor.execute("SELECT img_id, img_url FROM img WHERE label = '' ORDER BY RAND() LIMIT 1;")
            result = cursor.fetchone()  # Récupérer l'ID et l'URL d'une image
        
        cursor.close()
        conn.close()

        if result:
            if n == 3:
                image_ids = [row[0] for row in result]
                image_urls = [row[1] for row in result]
                return image_ids, image_urls  # Retourner les IDs et les URLs des images
            else:
                image_id, image_url = result
                return image_id, image_url  # Retourner l'ID et l'URL de l'image
        else:
            return None, None  # Retourner None si aucune image n'est trouvée

    except Exception as e:
        print("Erreur lors de la récupération de l'image depuis la base de données :", e)
        return None, None  # Retourner None en cas d'erreur
    