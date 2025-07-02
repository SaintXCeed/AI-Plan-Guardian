import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_streamlit_app():
    """Run the Streamlit application"""
    print("Starting Streamlit application...")
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        try:
            install_requirements()
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            sys.exit(1)
    
    # Run the Streamlit app
    run_streamlit_app()
