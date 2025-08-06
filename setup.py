#!/usr/bin/env python
"""
Setup script for Real Estate Village Comparison Django Application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Environment variables for Real Estate Analyzer\n")
            f.write("# Replace with your actual OpenAI API key\n")
            f.write("OPENAI_API_KEY=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")
        print("✅ .env file created. Please add your OpenAI API key.")
    else:
        print("✅ .env file already exists")

def check_data_file():
    """Check if SampleR.xlsx exists"""
    data_file = Path("SampleR.xlsx")
    if not data_file.exists():
        print("⚠️  SampleR.xlsx not found in project root")
        print("   Please add your data file to continue")
        return False
    print("✅ SampleR.xlsx found")
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_migrations():
    """Run Django migrations"""
    print("🗄️  Running Django migrations...")
    try:
        subprocess.check_call([sys.executable, "manage.py", "migrate"])
        print("✅ Migrations completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run migrations: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Real Estate Village Comparison Django Application")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Check data file
    data_exists = check_data_file()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Run migrations
    if not run_migrations():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    
    if not data_exists:
        print("\n⚠️  IMPORTANT: Add your SampleR.xlsx file to the project root")
    
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    if not data_exists:
        print("2. Add SampleR.xlsx to the project root")
    print("3. Run: python manage.py runserver")
    print("4. Open: http://127.0.0.1:8000/")
    
    print("\n🎉 Happy analyzing!")

if __name__ == "__main__":
    main() 