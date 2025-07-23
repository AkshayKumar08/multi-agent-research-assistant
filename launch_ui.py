#!/usr/bin/env python3
"""
UI Demo Launcher for Multi-Agent Research Assistant

This script provides options to launch different UI interfaces
and validates the setup before launching.
"""

import sys
import subprocess
from pathlib import Path
import importlib.util

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


def check_dependencies():
    """Check if required UI dependencies are installed."""
    print("🔍 Checking UI dependencies...")
    
    dependencies = {
        'streamlit': 'streamlit>=1.30.0',
        'plotly': 'plotly>=5.15.0',
        'pandas': 'pandas>=2.0.0',
        'matplotlib': 'matplotlib>=3.7.0'
    }
    
    missing = []
    available = []
    
    for dep_name, dep_spec in dependencies.items():
        spec = importlib.util.find_spec(dep_name)
        if spec is None:
            missing.append(dep_spec)
            print(f"❌ {dep_name} - Not installed")
        else:
            available.append(dep_name)
            print(f"✅ {dep_name} - Available")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False, available
    else:
        print("\n✅ All UI dependencies available!")
        return True, available


def check_backend_status():
    """Check if backend services are available."""
    print("\n🔍 Checking backend services...")
    
    try:
        from src.tools.ollama_client import OllamaClient
        client = OllamaClient()
        
        if client.is_available():
            models = client.list_models()
            print(f"✅ Ollama: Available with {len(models)} models")
            return True
        else:
            print("⚠️  Ollama: Server not responding")
            print("   Start with: ollama serve")
            return False
            
    except Exception as e:
        print(f"❌ Ollama: Error - {e}")
        return False


def launch_streamlit():
    """Launch the Streamlit interface."""
    print("\n🚀 Launching Streamlit interface...")
    try:
        # Use the custom launcher to suppress warnings
        subprocess.run([sys.executable, "run_streamlit.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        print("Trying direct launch...")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "ui_streamlit.py", "--server.headless", "true"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Direct launch also failed")
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface closed by user")


def install_ui_dependencies():
    """Install UI dependencies."""
    print("\n📦 Installing UI dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "streamlit>=1.30.0", "plotly>=5.15.0", "matplotlib>=3.7.0"
        ], check=True)
        print("✅ UI dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def main():
    """Main launcher function."""
    print("🤖 Multi-Agent Research Assistant - UI Launcher")
    print("=" * 55)
    
    # Check dependencies
    deps_ok, available = check_dependencies()
    backend_ok = check_backend_status()
    
    print("\n📋 System Status:")
    print(f"   UI Dependencies: {'✅ Ready' if deps_ok else '❌ Missing'}")
    print(f"   Backend Services: {'✅ Ready' if backend_ok else '⚠️  Limited'}")
    
    if not deps_ok:
        print("\n❓ Would you like to install missing UI dependencies?")
        choice = input("Enter 'y' to install, or any other key to skip: ").strip().lower()
        if choice == 'y':
            if install_ui_dependencies():
                deps_ok = True
            else:
                print("❌ Installation failed. Please install manually.")
    
    if not deps_ok:
        print("\n❌ Cannot launch UI without required dependencies.")
        print("Install manually with: pip install -r requirements.txt")
        return
    
    if not backend_ok:
        print("\n⚠️  Warning: Backend services limited. Some features may not work.")
        print("For full functionality, start Ollama with: ollama serve")
    
    # Interface selection
    print("\n🌐 Streamlit Interface:")
    print("1. 📊 Launch Streamlit Interface")
    print("2. ❌ Exit")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        if 'streamlit' in available:
            launch_streamlit()
        else:
            print("❌ Streamlit not available")
    elif choice == "2":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
