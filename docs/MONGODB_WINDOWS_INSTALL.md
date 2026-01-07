# Quick MongoDB Setup for Windows (Without Docker)

## Option 1: Install via Chocolatey (Fastest)

If you have Chocolatey installed:
```powershell
choco install mongodb
```

## Option 2: Manual Download & Install (Recommended)

1. **Download MongoDB Community Server:**
   - Go to: https://www.mongodb.com/try/download/community
   - Select: Windows x64
   - Click: Download (MSI)

2. **Install:**
   - Run the downloaded `.msi` file
   - Choose "Complete" installation
   - Select "Install MongoDB as a Service"
   - Use default data directory: `C:\Program Files\MongoDB\Server\8.2\data\`

3. **Verify Installation:**
   ```powershell
   # Check if service is running
   Get-Service MongoDB
   
   # If not running, start it
   Start-Service MongoDB
   ```

## Option 3: Portable MongoDB (No Installation Required)

For a quick test without installing:

1. **Download MongoDB ZIP:**
   ```powershell
   # Create MongoDB directory
   mkdir C:\mongodb
   cd C:\mongodb
   
   # Download MongoDB portable (replace URL with latest version)
   # Go to: https://www.mongodb.com/try/download/community
   # Download the ZIP version
   ```

2. **Extract and Run:**
   ```powershell
   # Extract the ZIP file to C:\mongodb
   
   # Create data directory
   mkdir C:\mongodb\data
   
   # Start MongoDB
   C:\mongodb\bin\mongod.exe --dbpath C:\mongodb\data
   ```

## Option 4: Use Our Installation Script

We've created a helper script:

```powershell
# Run from project root
.\install_mongodb.ps1
```

This will:
- Check if MongoDB is installed
- Install via winget if not present
- Start the MongoDB service
- Test the connection

## After Installation

### 1. Verify MongoDB is Running

```powershell
# Check service status
Get-Service MongoDB

# Test connection
.\activate_env.ps1
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017/'); client.admin.command('ping'); print('Connected!')"
```

### 2. Run the Demo

```powershell
.\activate_env.ps1
python scripts\demo_mongodb_storage.py
```

### 3. View Your Data (Optional)

Install MongoDB Compass (GUI tool):
```powershell
winget install MongoDB.Compass.Full
```

Or use the MongoDB Shell:
```powershell
winget install MongoDB.Shell
mongosh
```

## Quick Test Without MongoDB

If you want to test the chatbot without MongoDB:

The chatbot works perfectly fine without MongoDB! MongoDB is optional for:
- Storing conversation history
- User profiles
- Intent analytics

You can run all chatbot features without it:
```powershell
.\activate_env.ps1
python scripts\demo_therapy_chatbot.py
python scripts\day4_demo.py
```

## Troubleshooting

### "MongoDB service won't start"
```powershell
# Check if port 27017 is in use
netstat -ano | findstr :27017

# Restart the service
Restart-Service MongoDB
```

### "Connection refused"
- Make sure MongoDB service is running
- Check Windows Firewall isn't blocking port 27017
- Verify MongoDB is listening on localhost:27017

### Need Help?
Run our automated installer:
```powershell
.\install_mongodb.ps1
```

It will detect issues and guide you through the fix.
