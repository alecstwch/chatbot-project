# MongoDB Installation Script for Windows (No Docker Required)
# Run this script if MongoDB is not installed

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  MongoDB Installation for Windows" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if MongoDB is already installed
$mongoService = Get-Service -Name MongoDB -ErrorAction SilentlyContinue

if ($mongoService) {
    Write-Host "MongoDB is already installed!" -ForegroundColor Green
    Write-Host "Service Status: $($mongoService.Status)" -ForegroundColor Yellow
    
    if ($mongoService.Status -ne "Running") {
        Write-Host "`nStarting MongoDB service..." -ForegroundColor Yellow
        Start-Service -Name MongoDB
        Write-Host "MongoDB started successfully!" -ForegroundColor Green
    }
} else {
    Write-Host "[INFO] MongoDB not found. Installing..." -ForegroundColor Yellow
    Write-Host "`nUsing Windows Package Manager (winget)...`n" -ForegroundColor Cyan
    
    # Install MongoDB using winget
    winget install MongoDB.Server --accept-package-agreements --accept-source-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nMongoDB installed successfully!" -ForegroundColor Green
        
        # Wait a moment for service to be created
        Start-Sleep -Seconds 5
        
        # Try to start the service
        Write-Host "`nStarting MongoDB service..." -ForegroundColor Yellow
        try {
            Start-Service -Name MongoDB -ErrorAction Stop
            Write-Host "MongoDB service started!" -ForegroundColor Green
        } catch {
            Write-Host "[WARN] Could not start MongoDB service automatically" -ForegroundColor Yellow
            Write-Host "You may need to start it manually or reboot" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`n[ERROR] MongoDB installation failed" -ForegroundColor Red
        Write-Host "Please install manually from: https://www.mongodb.com/try/download/community" -ForegroundColor Yellow
        exit 1
    }
}

# Test MongoDB connection
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Testing MongoDB Connection" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Activating Python environment..." -ForegroundColor Yellow
& .\activate_env.ps1

Write-Host "`nTesting connection to MongoDB..." -ForegroundColor Yellow
python -c @"
from pymongo import MongoClient
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print('MongoDB connection successful!')
    print('MongoDB is ready to use.')
except Exception as e:
    print(f'[ERROR] Could not connect: {e}')
    print('MongoDB may still be starting up. Wait a moment and try again.')
"@

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "MongoDB Details:" -ForegroundColor Cyan
Write-Host "  Connection: mongodb://localhost:27017/" -ForegroundColor White
Write-Host "  Database: chatbot_db" -ForegroundColor White
Write-Host "  Data Directory: C:\Program Files\MongoDB\Server\8.2\data\" -ForegroundColor White
Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Run demo: python scripts\demo_mongodb_storage.py" -ForegroundColor White
Write-Host "  2. View data: Install MongoDB Compass (GUI)" -ForegroundColor White
Write-Host "     winget install MongoDB.Compass.Full" -ForegroundColor Gray
Write-Host ""
