# MongoDB Setup Guide

## Quick Start with Docker (Recommended)

The easiest way to get MongoDB running:

```powershell
# Pull and run MongoDB
docker run -d -p 27017:27017 --name chatbot-mongodb mongo:latest

# Verify it's running
docker ps
```

## Alternative: Local Installation

1. Download MongoDB Community Edition from: https://www.mongodb.com/try/download/community
2. Install MongoDB (default settings)
3. MongoDB will run on `mongodb://localhost:27017/`

## Configuration

The chatbot automatically connects to MongoDB using these defaults:

```
MongoDB URI: mongodb://localhost:27017/
Database: chatbot_db
Collections:
  - conversations (stores chat sessions)
  - users (stores user profiles)
```

### Custom Configuration

Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=chatbot_db
MONGODB_MAX_POOL_SIZE=10
```

## Test Connection

```powershell
# Activate environment
.\activate_env.ps1

# Test MongoDB demo
python scripts\demo_mongodb_storage.py
```

## View Data

### Option 1: MongoDB Compass (GUI)
1. Download: https://www.mongodb.com/try/download/compass
2. Connect to: `mongodb://localhost:27017`
3. Browse `chatbot_db` database

### Option 2: MongoDB Shell
```bash
# Connect to MongoDB
mongosh

# Switch to chatbot database
use chatbot_db

# View conversations
db.conversations.find().pretty()

# View users
db.users.find().pretty()

# Count sessions per user
db.conversations.aggregate([
  { $group: { _id: "$user_name", count: { $sum: 1 } } }
])
```

## Data Structure

### Conversation Document
```json
{
  "session_id": "uuid-here",
  "user_name": "john_doe",
  "started_at": "2026-01-07T10:00:00Z",
  "ended_at": "2026-01-07T10:15:00Z",
  "messages": [
    {
      "role": "user",
      "content": "I feel anxious",
      "timestamp": "2026-01-07T10:01:00Z",
      "intent": "anxiety",
      "intent_confidence": 0.95,
      "metadata": {}
    },
    {
      "role": "bot",
      "content": "Tell me more about that",
      "timestamp": "2026-01-07T10:01:01Z",
      "metadata": {"model": "aiml"}
    }
  ],
  "metadata": {}
}
```

### User Profile Document
```json
{
  "user_name": "john_doe",
  "total_sessions": 5,
  "total_messages": 42,
  "primary_intents": ["anxiety", "stress", "depression"],
  "first_seen": "2026-01-01T10:00:00Z",
  "last_seen": "2026-01-07T10:00:00Z",
  "metadata": {}
}
```

## Troubleshooting

### "Failed to connect to MongoDB"
- Check if MongoDB is running: `docker ps` or check Windows Services
- Verify port 27017 is not blocked
- Check connection string in `.env` file

### Permission Errors
```powershell
# If using Docker on Windows, you may need to run as admin
# Or add your user to docker-users group
```

### Performance Tips
- MongoDB creates indexes automatically
- For large datasets, consider adding compound indexes
- Use MongoDB Atlas for cloud deployment

## Production Deployment

For production, use MongoDB Atlas (free tier available):

1. Create account at: https://www.mongodb.com/cloud/atlas
2. Create a free cluster
3. Get connection string (looks like: `mongodb+srv://user:pass@cluster.mongodb.net/`)
4. Update `.env`:
   ```env
   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/chatbot_db
   ```
