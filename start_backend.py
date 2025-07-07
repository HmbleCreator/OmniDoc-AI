#!/usr/bin/env python3
"""
Startup script for OmniDoc AI Backend
This script ensures proper Python path and imports
"""

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent
backend_dir = project_root / "backend"

# Add backend directory to Python path
sys.path.insert(0, str(backend_dir))

# Change to backend directory
os.chdir(backend_dir)

# Import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    from main import app
    
    print("🚀 Starting OmniDoc AI Backend...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[0]}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 