#!/usr/bin/env python3
"""Installation verification for comparison training"""

import subprocess
import sys

def install_packages():
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    "code/requirements.txt", "-q"])
    print("Installation complete")

def verify_install():
    print("\nVerifying installation...")
    packages = ['torch', 'transformers', 'datasets', 'accelerate',
                'peft', 'sklearn', 'pandas', 'numpy', 'openpyxl']
    all_good = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"   {pkg}")
        except ImportError as e:
            print(f"   MISSING {pkg}: {e}")
            all_good = False
    return all_good

if __name__ == "__main__":
    install_packages()
    if verify_install():
        print("\nAll packages installed successfully!")
        print("Run: python run_first.py")
    else:
        print("\nSome packages failed. Re-run: pip install -r code/requirements.txt")
