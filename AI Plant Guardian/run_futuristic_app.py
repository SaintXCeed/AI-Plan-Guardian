import subprocess
import sys
import os
import time

def print_banner():
    """Print futuristic banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🚀 AI PLANT GUARDIAN - QUANTUM INITIALIZATION 🚀         ║
    ║                                                              ║
    ║         Next-Generation Plant Disease Detection System       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def install_requirements():
    """Install required packages with futuristic progress"""
    print("🔄 Initializing Quantum Dependencies...")
    print("=" * 60)
    
    try:
        # Install packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Simulate quantum installation progress
        components = [
            "Neural Network Cores",
            "Quantum Processors", 
            "AI Vision Systems",
            "Holographic Interface",
            "Data Analysis Engine"
        ]
        
        for i, component in enumerate(components):
            print(f"⚡ Installing {component}...", end="")
            time.sleep(0.5)
            print(" ✅ ONLINE")
        
        print("\n🎯 All quantum systems initialized successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Quantum initialization failed: {e}")
        return False

def run_quantum_app():
    """Launch the quantum application"""
    print("\n" + "=" * 60)
    print("🚀 Launching AI Plant Guardian...")
    print("🌐 Quantum Interface URL: http://localhost:8501")
    print("🤖 Neural Networks: ONLINE")
    print("⚡ Quantum Cores: ACTIVE")
    print("=" * 60)
    
    try:
        subprocess.run([
            "streamlit", "run", "streamlit_app.py", 
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Quantum system shutdown initiated...")
        print("👋 AI Plant Guardian offline")
    except Exception as e:
        print(f"❌ System error: {e}")

def main():
    print_banner()
    
    # Check system requirements
    if not os.path.exists("requirements.txt"):
        print("❌ Quantum configuration file missing!")
        print("📝 Please ensure 'requirements.txt' is present")
        return
    
    # Initialize quantum systems
    if install_requirements():
        print("\n🎯 System Status: ALL SYSTEMS GO")
        input("\n🚀 Press ENTER to launch AI Plant Guardian...")
        run_quantum_app()
    else:
        print("❌ Quantum initialization failed")
        print("🔧 Please check system requirements and try again")

if __name__ == "__main__":
    main()
