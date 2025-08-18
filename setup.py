#!/usr/bin/env python3

import os
import subprocess

def run_command(command):
    """Run a shell command and return success status."""
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e}")
        return False

def main():
    print("Setting up Wealth Agent Chat Application...")
    print("=" * 50)
    print("Note: Make sure you're in a virtual environment!")
    print("If not, run: python3 -m venv venv && source venv/bin/activate")
    print()
    
    # Install dependencies
    print("Installing Python dependencies...")
    if not run_command("pip install --upgrade pip"):
        print("Failed to upgrade pip")
        return False
        
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install dependencies")
        return False
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("\n⚠️  Environment file not found!")
        print("Creating .env file from template...")
        
        if os.path.exists(".env.example"):
            run_command("cp .env.example .env")
            print("✅ Created .env file from template")
        else:
            print("❌ .env.example not found")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. (Optional) Add LangSmith API key for tracing")
    print("3. Run: python main.py")

if __name__ == "__main__":
    main()