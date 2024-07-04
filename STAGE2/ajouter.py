import hashlib
from BD import get_BD

def ajouter(nom, email, mdp, exp):
    conn = get_BD()
    if conn is None:
        return False
    cursor = conn.cursor()
    
    # Hacher le mot de passe
    mdp_hash = hashlib.sha256(mdp.encode()).hexdigest()
    
    query = "INSERT INTO clients(nom, email, mdp, exp) VALUES(%s, %s, %s, %s)"
    cursor.execute(query, (nom, email, mdp_hash, exp))
    conn.commit()
    cursor.close()
    conn.close()
    return True

def verif_valid(nom, email):
    conn = get_BD()
    if conn is None:
        return False
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM clients WHERE nom = %s OR email = %s"
    cursor.execute(query, (nom, email))
    result = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    if result > 0:
        return True
    else:
        return False

