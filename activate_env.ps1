# Chatbot Project - Quick Activation Script
# Double-click this file or run: .\activate_env.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CHATBOT PROJECT ENVIRONMENT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".\chatbot-env\Scripts\Activate.ps1") {
    Write-Host "Found chatbot-env virtual environment" -ForegroundColor Green
    Write-Host ""
    
    # Activate the environment
    Write-Host "Activating environment..." -ForegroundColor Yellow
    & .\chatbot-env\Scripts\Activate.ps1
    
    Write-Host ""
    Write-Host "Environment activated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Python version:" -ForegroundColor Cyan
    python --version
    
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Cyan
    Write-Host "  python test_environment.py  - Test all packages" -ForegroundColor White
    Write-Host "  jupyter notebook            - Start Jupyter" -ForegroundColor White
    Write-Host "  python                      - Start Python REPL" -ForegroundColor White
    Write-Host ""
    Write-Host "Ready to code!" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host "ERROR: chatbot-env not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run setup first:" -ForegroundColor Yellow
    Write-Host "  py -3.11 -m venv chatbot-env" -ForegroundColor White
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
