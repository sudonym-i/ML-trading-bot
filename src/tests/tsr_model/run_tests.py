#!/usr/bin/env python3
"""
Test runner script for TSR model tests.

This script provides a convenient way to run all TSR model tests with
various options and configurations.

Usage:
    python run_tests.py [options]
    
Examples:
    # Run all tests
    python run_tests.py
    
    # Run only unit tests
    python run_tests.py --unit
    
    # Run with coverage
    python run_tests.py --coverage
    
    # Run specific test file
    python run_tests.py test_model.py
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(args_list):
    """Run pytest with given arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args_list
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser(description="Run TSR model tests")
    
    # Test selection options
    parser.add_argument("files", nargs="*", help="Specific test files to run")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests") 
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--network", action="store_true", help="Include network tests")
    
    # Output options
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--html-cov", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    parser.add_argument("--tb", choices=["short", "long", "line", "no"], default="short", 
                       help="Traceback format")
    
    # Performance options  
    parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel (requires pytest-xdist)")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    
    # Debugging options
    parser.add_argument("--pdb", action="store_true", help="Drop into debugger on failures")
    parser.add_argument("--lf", action="store_true", help="Run last failed tests only")
    parser.add_argument("--ff", action="store_true", help="Run failed tests first")
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Add basic options
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")
    
    pytest_args.extend(["--tb", args.tb])
    pytest_args.extend(["--timeout", str(args.timeout)])
    
    # Add test selection markers
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.slow:
        markers.append("slow")
    if args.network:
        markers.append("network")
    
    if markers:
        pytest_args.extend(["-m", " or ".join(markers)])
    elif not args.slow and not args.network:
        # By default, exclude slow and network tests
        pytest_args.extend(["-m", "not slow and not network"])
    
    # Add coverage options
    if args.coverage or args.html_cov:
        pytest_args.extend(["--cov=../../tsr_model", "--cov-report=term-missing"])
        if args.html_cov:
            pytest_args.extend(["--cov-report=html"])
    
    # Add parallel execution
    if args.parallel:
        pytest_args.extend(["-n", str(args.parallel)])
    
    # Add debugging options
    if args.pdb:
        pytest_args.append("--pdb")
    if args.lf:
        pytest_args.append("--lf")
    if args.ff:
        pytest_args.append("--ff")
    
    # Add specific files or default to current directory
    if args.files:
        pytest_args.extend(args.files)
    else:
        pytest_args.append(".")
    
    # Run the tests
    result = run_pytest(pytest_args)
    
    # Print summary
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {result.returncode}")
        
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())