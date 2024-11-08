# ChromaDB Docker Setup and API Documentation
---------------------------------------------

### Prepared by Aadityaa (Meeeee) :))
## Table of Contents
1. [API Endpoints](#api-endpoints)
2. [Docker Commands Reference](#docker-commands)
3. [Common Scenarios](#common-scenarios)
4. [Troubleshooting](#troubleshooting)

## API Endpoints

### 1. Health Check
```http
GET http://localhost:5001/
```
**Response:**
```json
{
    "chroma_status": "connected",
    "status": "healthy"
}
```

### 2. Load Data
```http
POST http://localhost:5001/load-data
```
**Response:**
```json
{
    "message": "Data loaded successfully",
    "rows_processed": 1000,
    "status": "success"
}
```

### 3. Search Users
```http
POST http://localhost:5001/search
```
**Request Body:**
```json
{
    "query": "Architecture and Construction in Arunachal Pradesh",
    "top_k": 5
}
```

### 4. Add User
```http
POST http://localhost:5001/add-user
```
**Request Body:**
```json
{
    "user_id": "new_user_1",
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@example.com",
    "location": "Arunachal Pradesh",
    "about": "Software Engineer",
    "skills": "Python, JavaScript",
    "tags": "Technology",
    "phone": "1234567890",
    "major": "Computer Science"
}
```
**Response:**
```json
{
    "message": "User added successfully",
    "status": "success"
}
```

### 5. Delete User
```http
DELETE http://localhost:5001/delete-user/<user_id>
```
**Response:**
```json
{
    "message": "User <user_id> deleted successfully",
    "status": "success"
}
```

## Docker Commands

### Basic Operations

| Command | Description |
|---------|-------------|
| `docker-compose up` | Start containers |
| `docker-compose down` | Stop and remove containers |
| `docker-compose stop` | Pause containers |
| `docker-compose start` | Resume containers |
| `docker-compose up --build` | Rebuild and start containers |
| `docker-compose down -v` | Stop and remove everything including volumes |

### Container Management

#### Stop Containers
```bash
# Method 1: Using Ctrl+C
# Press Ctrl+C in the terminal where docker-compose is running

# Method 2: Using command
docker-compose stop    # This will stop containers without removing them
```

#### Remove Containers
```bash
# Without volumes
docker-compose down   

# With volumes (clears all data)
docker-compose down -v   
```

#### Start Containers
```bash
# Normal start
docker-compose up    

# Start with rebuild
docker-compose up --build
```

## Common Scenarios

### Code Changes

#### When changing Flask code (app.py):
```bash
docker-compose down
docker-compose up --build
```

#### When changing requirements.txt:
```bash
docker-compose down
docker-compose build --no-cache   # Forces rebuild without using cache
docker-compose up
```

### Troubleshooting

#### Full System Cleanup
```bash
docker-compose down -v
docker builder prune -f
docker-compose up --build
```

#### Status Checking
```bash
# View running containers
docker ps    

# View service status
docker-compose ps   
```

#### Log Viewing
```bash
# All logs
docker-compose logs    

# Flask app logs only
docker-compose logs flask_app   

# ChromaDB logs only
docker-compose logs chroma_server   
```

## Quick Reference

| Command | Purpose |
|---------|----------|
| `docker-compose stop` | Pause containers |
| `docker-compose start` | Resume containers |
| `docker-compose down` | Stop and remove containers |
| `docker-compose up` | Start containers |
| `docker-compose up --build` | Rebuild and start containers |
| `docker-compose down -v` | Stop and remove everything including volumes |# ChromaDB-flask-Docker
