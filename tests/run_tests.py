import subprocess
import sys
import argparse
from datetime import datetime


def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Financial Agent Tests')
    parser.add_argument('--all', action='store_true', help='Run all test suites')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"📈 Financial Agent Test Suite")
    print(f"{'='*60}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Default: run all tests
    if not any([args.unit, args.integration, args.coverage]):
        args.all = True
    
    # Base pytest command
    verbose_flag = "-vv" if args.verbose else "-v"
    base_cmd = f"pytest test_financial_agent.py {verbose_flag}"
    
    # Run unit tests
    if args.unit or args.all:
        cmd = f"{base_cmd} -m unit"
        if args.fast:
            cmd += " -m 'not slow'"
        results.append(("Unit Tests", run_command(cmd, "Running Unit Tests")))
    
    # Run integration tests
    if args.integration or args.all:
        cmd = f"{base_cmd} -m integration"
        results.append(("Integration Tests", run_command(cmd, "Running Integration Tests")))
    
    # Run all tests with coverage
    if args.coverage or args.all:
        cmd = f"{base_cmd} --cov=financial_agent --cov-report=term-missing --cov-report=html"
        if args.fast:
            cmd += " -m 'not slow'"
        results.append(("Coverage Report", run_command(cmd, "Running Tests with Coverage")))
    
    # Run standard tests if no specific flags
    if not args.unit and not args.integration and not args.coverage and args.all:
        cmd = base_cmd
        if args.fast:
            cmd += " -m 'not slow'"
        results.append(("All Tests", run_command(cmd, "Running All Tests")))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    print(f"{'='*60}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    print(f"\n{'='*60}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()