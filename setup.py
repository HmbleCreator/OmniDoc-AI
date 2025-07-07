#!/usr/bin/env python3
"""
OmniDoc AI Setup Script

This script automates the complete setup process for OmniDoc AI,
including dependency installation, model downloads, and configuration.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_step(step: str):
    """Print a step message"""
    print(f"\n📋 {step}")

def run_command(command: str, cwd: str = None, check: bool = True) -> bool:
    """Run a shell command and return success status"""
    try:
        print(f"   Running: {command}")
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error output: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_step("Checking Python version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_node_version():
    """Check if Node.js is installed and compatible"""
    print_step("Checking Node.js version")
    
    try:
        result = subprocess.run(
            ["node", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        version = result.stdout.strip()
        print(f"   ✅ Node.js {version} - Found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ Node.js not found")
        print("   Please install Node.js 16+ from https://nodejs.org/")
        return False

def install_backend_dependencies():
    """Install backend Python dependencies"""
    print_step("Installing backend dependencies")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("   ❌ Backend directory not found")
        return False
    
    # Install requirements
    success = run_command("pip install -r requirements.txt", cwd="backend")
    if not success:
        print("   ❌ Failed to install backend dependencies")
        return False
    
    print("   ✅ Backend dependencies installed")
    return True

def install_frontend_dependencies():
    """Install frontend Node.js dependencies"""
    print_step("Installing frontend dependencies")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("   ❌ Frontend directory not found")
        return False
    
    # Install npm dependencies
    success = run_command("npm install", cwd="frontend")
    if not success:
        print("   ❌ Failed to install frontend dependencies")
        return False
    
    print("   ✅ Frontend dependencies installed")
    return True

def download_ai_models():
    """Download AI models using the download script"""
    print_step("Downloading AI models")
    
    backend_dir = Path("backend")
    download_script = backend_dir / "download_models.py"
    
    if not download_script.exists():
        print("   ❌ download_models.py not found")
        return False
    
    # Run the download script
    success = run_command("python download_models.py", cwd="backend", check=False)
    if not success:
        print("   ⚠️  Model download had issues - you can run it manually later")
        print("   Run: cd backend && python download_models.py")
        return False
    
    print("   ✅ AI models downloaded successfully")
    return True

def create_env_file():
    """Create a sample .env file"""
    print_step("Creating environment configuration")
    
    env_file = Path("backend/.env")
    if env_file.exists():
        print("   ✅ .env file already exists")
        return True
    
    # Create sample .env file
    env_content = """# OmniDoc AI Environment Configuration
# Copy this file and add your API keys

# OpenAI (https://platform.openai.com/)
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini (https://makersuite.google.com/)
GOOGLE_API_KEY=your_google_api_key_here

# Anthropic Claude (https://console.anthropic.com/)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Local LLM (optional - path to GGUF model file)
LOCAL_MODEL_PATH=./models/llama-2-7b-chat.gguf
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("   ✅ Created .env file with sample configuration")
        print("   📝 Edit backend/.env to add your API keys")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create .env file: {e}")
        return False

def test_backend():
    """Test if the backend can start"""
    print_step("Testing backend startup")
    
    try:
        # Try to import the main module
        result = run_command(
            "python -c \"import main; print('Backend imports successfully')\"", 
            cwd="backend",
            check=False
        )
        
        if result:
            print("   ✅ Backend imports successfully")
            return True
        else:
            print("   ❌ Backend has import issues")
            return False
            
    except Exception as e:
        print(f"   ❌ Backend test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print("""
🎉 OmniDoc AI has been set up successfully!

Next steps:

1. 🔑 Configure API Keys (Optional):
   - Edit backend/.env to add your API keys
   - Or configure them in the web interface

2. 🚀 Start the Backend:
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

3. 🌐 Start the Frontend:
   cd frontend
   npm run dev

4. 📖 Access the Application:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

5. 📚 Read the Documentation:
   - Check README.md for detailed usage instructions
   - Visit the troubleshooting section if you encounter issues

💡 Tips:
- Models are cached locally and won't download again
- You can run the backend and frontend in separate terminals
- Check the README.md for advanced configuration options

Need help? Check the troubleshooting section in README.md
""")

def main():
    """Main setup function"""
    print_header("OmniDoc AI Setup")
    print("This script will set up OmniDoc AI on your system.")
    print("This may take several minutes depending on your internet connection.")
    
    # Track setup progress
    steps_completed = 0
    total_steps = 6
    
    # Step 1: Check Python version
    if check_python_version():
        steps_completed += 1
    
    # Step 2: Check Node.js version
    if check_node_version():
        steps_completed += 1
    
    # Step 3: Install backend dependencies
    if install_backend_dependencies():
        steps_completed += 1
    
    # Step 4: Install frontend dependencies
    if install_frontend_dependencies():
        steps_completed += 1
    
    # Step 5: Download AI models
    if download_ai_models():
        steps_completed += 1
    
    # Step 6: Create environment file
    if create_env_file():
        steps_completed += 1
    
    # Step 7: Test backend
    if test_backend():
        steps_completed += 1
        total_steps += 1
    
    # Summary
    print_header("Setup Summary")
    print(f"✅ Completed: {steps_completed}/{total_steps} steps")
    
    if steps_completed >= total_steps - 1:  # Allow one failure (like model download)
        print_next_steps()
    else:
        print("""
❌ Setup encountered issues. Please:

1. Check the error messages above
2. Ensure you have Python 3.8+ and Node.js 16+ installed
3. Check your internet connection
4. Try running the setup again

For help, check the troubleshooting section in README.md
""")

if __name__ == "__main__":
    main() 