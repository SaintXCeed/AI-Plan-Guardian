import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔄 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    print("🚀 Starting Plant Disease AI Detector...")
    print("🌐 The app will open in your default web browser")
    print("📱 Access URL: http://localhost:8501")
    
    try:
        subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    print("🌱 Plant Disease AI Detector Setup")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install requirements
    if install_requirements():
        print("\n" + "=" * 40)
        run_streamlit_app()
    else:
        print("❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
