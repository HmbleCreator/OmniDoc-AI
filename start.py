#!/usr/bin/env python3
"""
OmniDoc AI - Startup Script
Launches both frontend and backend servers
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Check if backend dependencies exist
    backend_path = Path("backend")
    if not backend_path.exists():
        print("❌ Backend directory not found")
        return False
    
    requirements_path = backend_path / "requirements.txt"
    if not requirements_path.exists():
        print("❌ Backend requirements.txt not found")
        return False
    
    # Check if frontend dependencies exist
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found")
        return False
    
    package_json_path = frontend_path / "package.json"
    if not package_json_path.exists():
        print("❌ Frontend package.json not found")
        return False
    
    print("✅ Dependencies check passed")
    return True

def install_backend_dependencies():
    """Install backend Python dependencies"""
    print("📦 Installing backend dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"
        ], check=True)
        print("✅ Backend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install backend dependencies: {e}")
        return False

def install_frontend_dependencies():
    """Install frontend Node.js dependencies"""
    print("📦 Installing frontend dependencies...")
    
    try:
        subprocess.run(["npm", "install"], cwd="frontend", check=True)
        print("✅ Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install frontend dependencies: {e}")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend server...")
    
    try:
        # Change to backend directory
        os.chdir("backend")
        
        # Start the server
        process = subprocess.Popen([
            sys.executable, "main.py"
        ])
        
        print("✅ Backend server started on http://localhost:8000")
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the React frontend development server"""
    print("🚀 Starting frontend server...")
    
    try:
        # Change to frontend directory
        os.chdir("frontend")
        
        # Start the development server
        process = subprocess.Popen([
            "npm", "run", "dev"
        ])
        
        print("✅ Frontend server started on http://localhost:5173")
        return process
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🚀 OmniDoc AI - Starting up...")
    print("=" * 50)
    
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Check dependencies
        if not check_dependencies():
            print("❌ Dependency check failed. Please fix the issues above.")
            return 1
        
        # Ask user if they want to install dependencies
        install_deps = input("Do you want to install dependencies? (y/n): ").lower().strip()
        
        if install_deps == 'y':
            if not install_backend_dependencies():
                return 1
            if not install_frontend_dependencies():
                return 1
        
        # Start backend
        backend_process = start_backend()
        if not backend_process:
            return 1
        
        # Wait a moment for backend to start
        time.sleep(2)
        
        # Return to original directory for frontend
        os.chdir(original_dir)
        
        # Start frontend
        frontend_process = start_frontend()
        if not frontend_process:
            backend_process.terminate()
            return 1
        
        print("\n" + "=" * 50)
        print("🎉 OmniDoc AI is now running!")
        print("📱 Frontend: http://localhost:5173")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all servers")
        print("=" * 50)
        
        # Wait for user to stop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            
            # Terminate processes
            if backend_process:
                backend_process.terminate()
            if frontend_process:
                frontend_process.terminate()
            
            print("✅ Servers stopped")
            return 0
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        return 1
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main()) 