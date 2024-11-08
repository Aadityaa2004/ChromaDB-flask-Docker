import pandas as pd
import chromadb
from flask import Flask, request, jsonify
from chromadb.utils import embedding_functions
import os
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# Initialize ChromaDB client
def init_chroma():
    """Initialize ChromaDB client with error handling"""
    try:
        client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", 8000))
        )
        client.heartbeat()
        logger.info("Successfully connected to ChromaDB")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")
        raise

# Initialize embedding function
def init_embedding():
    """Initialize embedding function with error handling"""
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {str(e)}")
        raise

try:
    client = init_chroma()
    embedding_function = init_embedding()
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
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
    
@app.route('/select-user/<user_id>', methods=['GET'])
def select_user(user_id):
    """Get a specific user's profile for search context"""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.get(
            ids=[user_id],
            include=['metadatas', 'documents']
        )
        
        if not results['ids']:
            return jsonify({
                "error": "User not found",
                "status": "error"
            }), 404
            
        return jsonify({
            "user_profile": results['metadatas'][0],
            "status": "success"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
    
@app.route('/dynamic-search', methods=['POST'])
def dynamic_search():
    """Dynamic search based on search type and user context"""
    try:
        data = request.json
        query_text = data.get('query', '')
        search_type = data.get('search_type', 'general')  # name, skills, tags, about, general
        context_user_id = data.get('context_user_id')  # ID of the user performing the search
        n_results = data.get('top_k', 5)
        
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Get context user's profile if provided
        context_user_profile = None
        if context_user_id:
            context_results = collection.get(
                ids=[context_user_id],
                include=['metadatas']
            )
            if context_results['metadatas']:
                context_user_profile = context_results['metadatas'][0]
        
        # Adjust search based on type
        if search_type == 'name':
            # Exact name matching
            results = collection.query(
                query_texts=[query_text],
                n_results=100  # Get more results initially for filtering
            )
            
            # Filter results for exact name matches
            filtered_results = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                full_name = f"{metadata['first_name']} {metadata['middle_name']} {metadata['last_name']}".lower()
                if query_text.lower() in full_name:
                    filtered_results.append({
                        "id": results['ids'][0][i],
                        "metadata": metadata,
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })
            
            return jsonify({
                "results": filtered_results[:n_results],
                "status": "success"
            }), 200
            
        elif search_type in ['skills', 'tags', 'about']:
            # Enhanced search using context user's profile
            enhanced_query = query_text
            if context_user_profile:
                # Combine search query with relevant context from user's profile
                if search_type == 'skills':
                    enhanced_query = f"{query_text} {context_user_profile.get('skills', '')}"
                elif search_type == 'tags':
                    enhanced_query = f"{query_text} {context_user_profile.get('tags', '')}"
                elif search_type == 'about':
                    enhanced_query = f"{query_text} {context_user_profile.get('about', '')}"
            
            results = collection.query(
                query_texts=[enhanced_query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                if results['ids'][0][i] != context_user_id:  # Exclude the context user
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
            
        else:  # general search
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
    
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify API is working"""
    return jsonify({"status": "API is running"}), 200

@app.route('/database-stats', methods=['GET'])
def get_database_stats():
    """Return statistics about the database"""
    logger.debug("Accessing database-stats endpoint")
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        logger.debug(f"Retrieved collection: {COLLECTION_NAME}")
        
        # Get collection stats
        all_results = collection.get()
        total_rows = len(all_results['ids'])
        logger.info(f"Total rows in database: {total_rows}")
        
        return jsonify({
            "total_rows": total_rows,
            "collection_name": COLLECTION_NAME,
            "status": "success"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in database-stats: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/random-users', methods=['GET'])
def get_random_users():
    """Return 10 random users from the database"""
    logger.debug("Accessing random-users endpoint")
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Get all user IDs
        all_results = collection.get()
        all_ids = all_results['ids']
        
        # Select 10 random IDs
        sample_size = min(10, len(all_ids))
        random_ids = random.sample(all_ids, sample_size)
        
        # Get the selected users
        results = collection.get(
            ids=random_ids,
            include=['metadatas', 'documents']
        )
        
        formatted_results = []
        for i in range(len(results['ids'])):
            result = {
                "id": results['ids'][i],
                "metadata": results['metadatas'][i],
                "document": results['documents'][i]
            }
            formatted_results.append(result)
        
        return jsonify({
            "results": formatted_results,
            "status": "success"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in random-users: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


if __name__ == '__main__':
    # Add debug output for registered routes
    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"{rule.endpoint}: {rule.methods} {rule}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)