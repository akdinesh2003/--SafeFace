#!/usr/bin/env python3
"""
Face Mask Detection Launcher
Choose which detection method to run
"""

import os
import sys
import subprocess

def print_banner():
    """Print project banner"""
    print("=" * 60)
    print("           FACE MASK DETECTION PROJECT")
    print("=" * 60)
    print()

def print_menu():
    """Print menu options"""
    print("Choose a detection method:")
    print("1. Demo Version (Recommended for testing)")
    print("   - Fast, works immediately")
    print("   - Uses color and texture analysis")
    print("   - No model training required")
    print()
    print("2. Basic Detection")
    print("   - Simple CNN model")
    print("   - Moderate accuracy")
    print("   - Requires model creation")
    print()
    print("3. Advanced Detection")
    print("   - Deep CNN with batch normalization")
    print("   - Highest accuracy")
    print("   - Slower processing")
    print()
    print("4. Install Dependencies")
    print("   - Install required Python packages")
    print()
    print("5. Exit")
    print()

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Error installing dependencies. Please install manually:")
        print("pip install -r requirements.txt")
    except FileNotFoundError:
        print("requirements.txt not found. Please ensure you're in the correct directory.")

def run_detector(script_name):
    """Run the selected detector script"""
    if not os.path.exists(script_name):
        print(f"Error: {script_name} not found!")
        return
    
    print(f"Starting {script_name}...")
    print("Press 'q' to quit the detection")
    print()
    
    try:
        subprocess.run([sys.executable, script_name])
    except KeyboardInterrupt:
        print("\nDetection stopped by user.")
    except Exception as e:
        print(f"Error running {script_name}: {e}")

def main():
    """Main launcher function"""
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                run_detector("demo_mask_detector.py")
            elif choice == "2":
                run_detector("face_mask_detection.py")
            elif choice == "3":
                run_detector("advanced_mask_detector.py")
            elif choice == "4":
                install_dependencies()
                input("\nPress Enter to continue...")
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
