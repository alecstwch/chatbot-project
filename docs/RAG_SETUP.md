# RAG Chatbot Setup Guide

This guide explains how to set up and run the RAG (Retrieval-Augmented Generation) enhanced chatbot with Qdrant vector database.

## What is RAG?

RAG enables the chatbot to:
- **Remember** past conversations semantically
- **Retrieve** relevant context when you ask questions
- **Augment** responses with historical information

Example:
```
Session 1: "My name is Alex and I had depression last year"
Session 2: "I'm feeling anxious today"
→ Bot recalls the depression history and responds with context
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG CHATBOT                              │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Gemini    │    │   Qdrant    │    │  Sentence       │ │
│  │   API       │    │   Vector DB │    │  Transformer    │ │
│  │  (Generate) │    │  (Retrieve) │    │  (Embed)        │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Gemini API Key** - Get from https://aistudio.google.com/
2. **Podman** (for Qdrant server) or use local storage -> or Docker
3. **Python 3.12+** with virtual environment

## Quick Start (Local Storage - No Server)

The simplest way - uses local file storage for Qdrant:

```bash
# 1. Activate environment
cd /Users/aursu/projects/chatbot-project
source venv/bin/activate

# 2. Install dependencies
pip install qdrant-client sentence-transformers

# 3. Set your API key
echo "GEMINI_API_KEY=your_key_here" >> .env

# 4. Run the chatbot
python -m src.interfaces.cli.rag_chatbot_cli --user-id myname
```

## Setup with Qdrant Server (Podman)

For better performance and persistence:

### Step 1: Start Podman Machine (macOS)

```bash
# Initialize Podman (first time only)
podman machine init

# Start the VM
podman machine start
```

### Step 2: Run Qdrant Container

```bash
# Create persistent volume
podman volume create qdrant_storage

# Run Qdrant
podman run -d \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage:z \
  --name qdrant \
  qdrant/qdrant
```

### Step 3: Verify Qdrant is Running

```bash
# Check container
podman ps

# Test API
curl http://localhost:6333/collections
```

### Step 4: Configure Environment

Create/update `.env` file:

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Server
QDRANT_USE_LOCAL_STORAGE=false
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional: Collection name
QDRANT_COLLECTION_NAME=conversations
```

### Step 5: Run RAG Chatbot

```bash
# With server
python -m src.interfaces.cli.rag_chatbot_cli \
  --user-id alex \
  --user-name "Alex" \
  --qdrant-server

# With therapy mode
python -m src.interfaces.cli.rag_chatbot_cli \
  --user-id alex \
  --therapy \
  --qdrant-server
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--user-id` | Unique identifier for memory persistence |
| `--user-name` | Display name for personalization |
| `--therapy` | Enable therapy/mental health mode |
| `--qdrant-server` | Use Qdrant server instead of local storage |
| `--qdrant-host` | Qdrant server host (default: localhost) |
| `--qdrant-port` | Qdrant server port (default: 6333) |

## Interactive Commands

While chatting, you can use these commands:

| Command | Description |
|---------|-------------|
| `help` | Show all commands |
| `search` | Search past conversations by topic |
| `data` | View your stored conversations |
| `alldata` | View ALL data from all users |
| `memory` | View memory statistics |
| `reset` | Clear current conversation (keeps memory) |
| `forget` | Delete ALL stored data (GDPR) |
| `history` | Show current session history |
| `clear` | Clear the screen |
| `quit` | Exit the chatbot |

### Commands Summary (Detailed)

#### `reset` - Start a New Conversation
Clears the current conversation but **preserves all memories** in Qdrant.
```
You: reset
✅ Conversation cleared. Memory preserved.

You: Hello
Bot: Hi Alex! How can I help you today?
     ↑ Still remembers your name from past conversations!
```

#### `forget` - Delete All Your Data
Permanently deletes ALL your stored conversations from Qdrant (GDPR compliance).
```
You: forget
⚠️ This will delete ALL your stored conversations. Type 'DELETE' to confirm: DELETE
✅ All your data has been deleted.
```

#### `search` - Search Past Conversations
Semantically search through your conversation history.
```
You: search
🔍 Search your conversation history
Enter topic to search (e.g., 'anxiety', 'work stress'): depression

📚 Found 3 relevant messages:
  1. [You] (87% match): "I had depression last year..."
  2. [Bot] (82% match): "I'm sorry to hear about your struggle..."
  3. [You] (71% match): "Sometimes I feel really down..."
```

#### `memory` - View Memory Statistics
Shows how many messages are stored for your user.
```
You: memory
==================================================
Memory Statistics
==================================================
  Status: ✅ Enabled
  User ID: alex
  Stored Messages: 42
  Collection: conversations
==================================================
```

#### `history` - View Current Session
Shows the conversation history from the current session only.
```
You: history
==================================================
Current Conversation
==================================================
[1] You: My name is Alex
    Bot: Nice to meet you, Alex!
[2] You: I'm feeling anxious
    Bot: I understand. Can you tell me more?
==================================================
```

#### `data` - View All Stored Conversations
Shows ALL messages stored in Qdrant, grouped by session.
```
You: data
======================================================================
📚 Stored Conversations for: alex
   Total messages: 12
======================================================================

┌─ Session: abc123... (4 messages)
│
│  [01/10 10:00] 👤 You: My name is Alex
│  [01/10 10:00] 🤖 Bot: Nice to meet you, Alex!
│  [01/10 10:01] 👤 You: I had depression last year
│  [01/10 10:01] 🤖 Bot: I'm sorry to hear that...
│
└──────────────────────────────────────────────────

┌─ Session: def456... (8 messages)
│
│  [01/11 14:30] 👤 You: I'm feeling better today
│  [01/11 14:30] 🤖 Bot: That's wonderful to hear!
│  ...
│
└──────────────────────────────────────────────────

💡 Tip: Use 'search' to find specific topics
======================================================================
```

#### `alldata` - View ALL Data (All Users)
Shows ALL messages from ALL users stored in Qdrant. Useful for learning/debugging.
```
You: alldata

======================================================================
📚 ALL Data in Qdrant (Admin View)
   Total messages: 24
   Total users: 3
======================================================================

──────────────────────────────────────────────────────────────────────
👤 USER: alex (12 messages)
──────────────────────────────────────────────────────────────────────

  ┌─ Session: abc123... (6 messages)
  │
  │  [01/10 10:00] 💬 user: My name is Alex
  │  [01/10 10:00] 🤖 assistant: Nice to meet you, Alex!
  │  [01/10 10:01] 💬 user: I had depression last year
  │  [01/10 10:01] 🤖 assistant: I'm sorry to hear that...
  │
  └──────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────
👤 USER: claudia (8 messages)
──────────────────────────────────────────────────────────────────────

  ┌─ Session: def456... (8 messages)
  │
  │  [01/10 14:30] 💬 user: I am hungry. what to do
  │  [01/10 14:30] 🤖 assistant: Let's get you sorted...
  │
  └──────────────────────────────────────────────────

======================================================================
💡 This shows ALL data from ALL users in Qdrant
======================================================================
```

#### `quit` / `exit` / `bye` - Exit the Chatbot
Exits the chatbot. **All memories are preserved** in Qdrant for next time.
```
You: quit
🤖 Goodbye! Your memories are saved for next time!
```

### Data Persistence Summary

| Action | Current Conversation | RAG Memory (Qdrant) |
|--------|---------------------|---------------------|
| `reset` | ❌ Cleared | ✅ Preserved |
| `quit` | ❌ Cleared | ✅ Preserved |
| `forget` | ❌ Cleared | ❌ **Deleted** |
| Close terminal | ❌ Cleared | ✅ Preserved |
| Restart computer | ❌ Cleared | ✅ Preserved |

## Example Session

```
🧠 RAG-Enhanced Conversational Bot
==================================================
User: alex
📚 Memory: 42 stored messages
--------------------------------------------------

You: My name is Sarah and I work as a nurse

Bot: Nice to meet you, Sarah! Being a nurse must be both 
rewarding and challenging. How can I help you today?

[0.89s | 45 tokens | 📚 0 memories used]

You: I've been feeling burned out lately

Bot: I understand, Sarah. Healthcare workers often experience 
burnout, especially given the demanding nature of nursing. 
Would you like to talk about what's been contributing to 
these feelings?

[1.2s | 52 tokens | 📚 2 memories used]
```

## Podman Commands Reference

```bash
# Start Qdrant
podman start qdrant

# Stop Qdrant
podman stop qdrant

# View logs
podman logs qdrant

# Remove container
podman rm qdrant

# List volumes
podman volume ls

# Remove volume (deletes all data!)
podman volume rm qdrant_storage
```

## Troubleshooting

### Podman machine not starting
```bash
podman machine stop
podman machine rm
podman machine init
podman machine start
```

### Qdrant connection refused
```bash
# Check if container is running
podman ps -a

# Restart container
podman restart qdrant

# Check logs
podman logs qdrant
```

### Memory not persisting
- For local storage: Check `./data/qdrant_db` exists
- For server: Ensure volume is mounted correctly

### Slow first response
- First message loads the embedding model (~300MB)
- Subsequent messages are faster

## Files Created

```
src/infrastructure/
├── config/
│   └── qdrant_settings.py     # Qdrant configuration
└── memory/
    ├── __init__.py
    ├── rag_memory_service.py  # Vector storage & retrieval
    └── rag_prompt_builder.py  # Context-aware prompts

src/infrastructure/ml/chatbots/
└── rag_chatbot.py             # RAG-enhanced chatbot

src/interfaces/cli/
└── rag_chatbot_cli.py         # Interactive CLI

scripts/
└── demo_rag_chatbot.py        # Demo script
```

## Next Steps

1. **Run the demo**: `python scripts/demo_rag_chatbot.py`
2. **Start chatting**: `python -m src.interfaces.cli.rag_chatbot_cli`
3. **View data**: Connect to Qdrant at http://localhost:6333/dashboard

