import pandas as pd
import chromadb
from flask import Flask, request, jsonify
from chromadb.utils import embedding_functions
import os
import time

app = Flask(__name__)

# Constants
COLLECTION_NAME = "user_profiles"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CSV_PATH = "sample1.csv"

def wait_for_chroma():
    """Wait for ChromaDB to be ready"""
    max_retries = 30
    retry_delay = 1  # seconds
    
    for i in range(max_retries):
        try:
            client = chromadb.HttpClient(
                host=os.getenv("CHROMA_HOST", "localhost"),
                port=int(os.getenv("CHROMA_PORT", 8000))
            )
            client.heartbeat()
            print("Successfully connected to ChromaDB")
            return client
        except Exception as e:
            print(f"Waiting for ChromaDB to be ready... (Attempt {i+1}/{max_retries})")
            if i < max_retries - 1:
                time.sleep(retry_delay)
    raise Exception("Could not connect to ChromaDB")

# Initialize ChromaDB client
print("Initializing ChromaDB client...")
client = wait_for_chroma()

# Initialize embedding function
print("Initializing embedding function...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME
)

def create_collection():
    """Create or get ChromaDB collection with error handling"""
    try:
        # Delete existing collection if it exists
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception as e:
            print(f"No existing collection to delete: {str(e)}")
            
        # Create new collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"Created new collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        raise

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "chroma_status": "connected" if client.heartbeat() else "disconnected"
    })

@app.route('/load-data', methods=['POST'])
def load_data():
    """Load data from CSV into ChromaDB"""
    try:
        # Read CSV
        df = pd.read_csv(CSV_PATH)
        collection = create_collection()
        print(f"Successfully read CSV with {len(df)} rows")

        # Create collection
        collection = create_collection()

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        # Make IDs unique by adding index if duplicates exist
        seen_ids = {}
        
        for idx, row in df.iterrows():
            # Create document text
            doc_text = f"{row.get('About', '')} {row.get('Skills', '')} {row.get('Tags', '')}".strip()
            
            # Make user_id unique
            base_id = str(row['User ID'])
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                user_id = f"{base_id}_{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0
                user_id = base_id

            # Prepare metadata
            metadata = {
                "user_id": user_id,
                "first_name": str(row['First Name']),
                "middle_name": str(row['Middle Name']),
                "last_name": str(row['Last Name']),
                "email": str(row['Email']),
                "location": str(row['Location']),
                "phone": str(row['Phone Number']),
                "about": str(row['About']),
                "major": str(row['Major']),
                "skills": str(row['Skills']),
                "tags": str(row['Tags'])
            }

            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(user_id)

        # Add data to collection in smaller batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )

        return jsonify({
            "message": "Data loaded successfully",
            "rows_processed": len(df),
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error",
            "details": {
                "csv_path": os.path.abspath(CSV_PATH),
                "current_directory": os.getcwd(),
                "exists": os.path.exists(CSV_PATH),
                "files_in_directory": os.listdir(os.getcwd())
            }
        }), 500

@app.route('/search', methods=['POST'])
def search():
    """Search for users based on query"""
    try:
        data = request.json
        query_text = data.get('query', '')
        n_results = data.get('top_k', 5)
        
        collection = client.get_collection(name=COLLECTION_NAME)

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "id": results['ids'][0][i],
                "metadata": results['metadatas'][0][i],
                "document": results['documents'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results.append(result)

        return jsonify({
            "results": formatted_results,
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/add-user', methods=['POST'])
def add_user():
    """Add a new user"""
    try:
        data = request.json
        collection = client.get_collection(name=COLLECTION_NAME)

        # Create document text
        doc_text = f"{data.get('about', '')} {data.get('skills', '')} {data.get('tags', '')}".strip()

        # Add user to collection
        collection.add(
            documents=[doc_text],
            metadatas=[data],
            ids=[data['user_id']]
        )

        return jsonify({
            "message": "User added successfully",
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/delete-user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        
        collection.delete(ids=[user_id])
        
        return jsonify({
            "message": f"User {user_id} deleted successfully",
            "status": "success"
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)