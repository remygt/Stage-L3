import hashlib
from BD import get_BD


def verif_conn(nom, mdp):
    conn = get_BD()
    if conn is None:
        return False, False  # Return a tuple with False for both login success and expert status
    cursor = conn.cursor()

    # Hash the password
    hashed_mdp = hashlib.sha256(mdp.encode()).hexdigest()

    query = "SELECT exp FROM clients WHERE nom = %s AND mdp = %s"
    cursor.execute(query, (nom, hashed_mdp))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        is_expert = result[0] == 1  # Assuming `exp` is stored as 1 for True and 0 for False
        return True, is_expert
    else:
        return False, False