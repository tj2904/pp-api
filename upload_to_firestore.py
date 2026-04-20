import json
import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import RetryError, AlreadyExists, DeadlineExceeded
from google.api_core.retry import Retry
import time

# Initialize Firebase Admin SDK
cred = credentials.Certificate(
    'positive-press-api-firebase-adminsdk-service-account-key.json')
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Load your JSON data
with open('basicVaderScoredNews.json') as json_file:
    data = json.load(json_file)


def upload_json_to_firestore(data, collection_name):
    """Function to upload JSON data to Firestore with retry mechanism"""
    # Reference to your Firestore collection
    collection_ref = db.collection(collection_name)

    # Retry configuration
    retry = Retry(initial=1.0, maximum=60.0, multiplier=2.0, deadline=300.0)

    # Track successfully uploaded documents
    uploaded_docs = set()

    for doc in data:
        # Rename the 'id' field to 'itemUrl'
        if 'id' in doc:
            doc['itemUrl'] = doc.pop('id')

        # Use the add method to generate a new ID for each document
        while True:
            try:
                doc_ref = collection_ref.add(doc, retry=retry)
                print(
                    f"Document with ID {doc_ref[1].id} uploaded successfully.")
                uploaded_docs.add(doc_ref[1].id)
                break
            except RetryError as e:
                print(f"RetryError: {e}. Retrying...")
                time.sleep(5)
            except DeadlineExceeded as e:
                print(f"DeadlineExceeded: {e}. Retrying...")
                time.sleep(5)
            except AlreadyExists:
                print(f"Document already exists. Skipping...")
                break

    return uploaded_docs


if __name__ == "__main__":
    uploaded_docs = upload_json_to_firestore(data, 'basicVaderScoredNews')
    print("Data upload complete.")
    print(f"Successfully uploaded documents: {uploaded_docs}")
