
#!/usr/bin/env python3
"""
Main script to reproduce all paper results.

Usage:
    python reproduce_paper_results.py --config configs/paper_experiments.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Import our implementations
sys.path.append('models')
sys.path.append('experiments')

from accuracy_evaluation import run_accuracy_experiments
from power_analysis import run_power_experiments  
from statistical_tests import run_significance_tests

def main():
    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--config', default='configs/paper_experiments.json')
    parser.add_argument('--output', default='results/')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level)
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    print("🚀 Starting paper results reproduction...")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {args.output}")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    print("\n📊 Running accuracy evaluation...")
    accuracy_results = run_accuracy_experiments(config)
    
    print("\n⚡ Running power analysis...")
    power_results = run_power_experiments(config)
    
    print("\n📈 Running statistical tests...")
    significance_results = run_significance_tests(accuracy_results, power_results)
    
    # Save results
    results = {
        'accuracy': accuracy_results,
        'power': power_results,
        'statistics': significance_results
    }
    
    with open(Path(args.output) / 'reproduction_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Paper results reproduction completed!")
    print(f"Results saved to: {args.output}/reproduction_results.json")

if __name__ == "__main__":
    main()
