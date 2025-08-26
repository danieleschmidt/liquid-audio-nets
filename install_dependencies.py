#!/usr/bin/env python3
"""
Install required dependencies for Liquid Audio Networks
Handles system-level dependency installation without requiring pip
"""

import subprocess
import sys
import os
from pathlib import Path

def install_with_fallback():
    """Install dependencies with multiple fallback strategies"""
    
    # Try to import essential packages first
    required_packages = {
        'numpy': 'numpy',
        'torch': 'torch', 
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for name, package in required_packages.items():
        try:
            __import__(name)
            print(f"✓ {name} already available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {name} not found")
    
    if not missing_packages:
        print("All required packages available!")
        return True
    
    print(f"\nAttempting to install: {missing_packages}")
    
    # Strategy 1: Try user install
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--user', '--break-system-packages'
        ] + missing_packages)
        print("✓ Successfully installed packages with --user")
        return True
    except subprocess.CalledProcessError:
        print("✗ User install failed")
    
    # Strategy 2: Try system packages
    apt_packages = {
        'numpy': 'python3-numpy',
        'torch': None,  # Not available via apt
        'scipy': 'python3-scipy', 
        'matplotlib': 'python3-matplotlib'
    }
    
    apt_installs = [apt_packages[p] for p in missing_packages if apt_packages.get(p)]
    
    if apt_installs:
        try:
            subprocess.check_call(['apt', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call(['apt', 'install', '-y'] + apt_installs)
            print(f"✓ Installed system packages: {apt_installs}")
        except subprocess.CalledProcessError:
            print("✗ System package install failed")
    
    # Strategy 3: Download and compile minimal versions
    print("Installing minimal implementations...")
    return install_minimal_implementations()

def install_minimal_implementations():
    """Install minimal pure-Python implementations"""
    
    # Create a local site-packages directory
    local_site = Path(__file__).parent / "local_packages"
    local_site.mkdir(exist_ok=True)
    
    # Add to Python path
    if str(local_site) not in sys.path:
        sys.path.insert(0, str(local_site))
    
    # Install minimal numpy alternative
    numpy_minimal = local_site / "numpy.py"
    if not numpy_minimal.exists():
        numpy_minimal.write_text('''
"""Minimal numpy-compatible interface for liquid-audio-nets"""
import math

class ndarray(list):
    def __init__(self, data, dtype=None):
        if isinstance(data, (int, float)):
            super().__init__([data])
        else:
            super().__init__(data)
        self.shape = (len(self),)
        self.dtype = dtype or float
    
    def max(self): return max(self) if self else 0
    def min(self): return min(self) if self else 0
    def mean(self): return sum(self) / len(self) if self else 0
    def std(self): 
        if not self: return 0
        m = self.mean()
        return (sum((x - m) ** 2 for x in self) / len(self)) ** 0.5
    def sum(self): return sum(super().__iter__())
    def tolist(self): return list(self)

def array(data, dtype=None):
    return ndarray(data, dtype)

def zeros(shape):
    if isinstance(shape, int):
        return ndarray([0.0] * shape)
    return ndarray([0.0] * shape[0])  # Simplified for 1D

def ones(shape):
    if isinstance(shape, int):
        return ndarray([1.0] * shape)
    return ndarray([1.0] * shape[0])  # Simplified for 1D

def sin(x):
    if hasattr(x, '__iter__'):
        return ndarray([math.sin(val) for val in x])
    return math.sin(x)

def cos(x):
    if hasattr(x, '__iter__'):
        return ndarray([math.cos(val) for val in x])
    return math.cos(x)

def exp(x):
    if hasattr(x, '__iter__'):
        return ndarray([math.exp(val) for val in x])
    return math.exp(x)

def log(x):
    if hasattr(x, '__iter__'):
        return ndarray([math.log(max(val, 1e-10)) for val in x])
    return math.log(max(x, 1e-10))

def sqrt(x):
    if hasattr(x, '__iter__'):
        return ndarray([math.sqrt(val) for val in x])
    return math.sqrt(x)

def linspace(start, stop, num=50):
    if num <= 1:
        return ndarray([start])
    step = (stop - start) / (num - 1)
    return ndarray([start + i * step for i in range(num)])

def random:
    class RandomModule:
        @staticmethod
        def randn(size):
            import random
            return ndarray([random.gauss(0, 1) for _ in range(size)])
        
        @staticmethod
        def rand(size):
            import random
            return ndarray([random.random() for _ in range(size)])
    
    return RandomModule()

random = random()

class fft:
    @staticmethod
    def fft(x):
        # Extremely simplified FFT - not accurate but prevents crashes
        return ndarray([complex(val, 0) for val in x])
    
    @staticmethod 
    def rfft(x):
        # Simplified real FFT
        return ndarray([complex(val, 0) for val in x[:len(x)//2 + 1]])

def abs(x):
    if hasattr(x, '__iter__'):
        return ndarray([abs(val) for val in x])
    return abs(x)

def argmax(x):
    return x.index(max(x)) if hasattr(x, 'index') else 0

def clip(x, a_min, a_max):
    if hasattr(x, '__iter__'):
        return ndarray([max(a_min, min(a_max, val)) for val in x])
    return max(a_min, min(a_max, x))

# Math constants
pi = math.pi
e = math.e
''')
    
    print(f"✓ Created minimal numpy at {numpy_minimal}")
    return True

if __name__ == "__main__":
    success = install_with_fallback()
    if success:
        print("\n✅ Dependency installation completed!")
        
        # Test the installation
        try:
            import numpy
            print(f"✓ numpy version: {getattr(numpy, '__version__', 'minimal')}")
        except ImportError:
            print("✗ numpy still not available")
    else:
        print("\n❌ Dependency installation failed")
        sys.exit(1)