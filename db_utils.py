global_db_client = None


def connect_db(account_key_path):
    global global_db_client
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import firestore

    if not global_db_client:
        cred = credentials.Certificate(account_key_path)
        firebase_admin.initialize_app(cred)
        db_client = firestore.client()
        global_db_client = db_client
    else:
        db_client = global_db_client

    return db_client


def upload_db(db_client, collection_name, content_dict):
    collection = db_client.collection(collection_name)
    for key, val in content_dict.items():
        doc_ref = collection.document(key)
        doc_ref.set(val)
