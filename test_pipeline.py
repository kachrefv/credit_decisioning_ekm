"""
Automated test execution pipeline for the Credithos EKM system.
Manages execution of all test suites with reporting and scheduling capabilities.
"""
import subprocess
import sys
import os
import time
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Any
import argparse


class TestPipeline:
    """Automated test execution pipeline."""
    
    def __init__(self, test_directory: str = "tests", reports_dir: str = "test_reports"):
        self.test_directory = Path(test_directory)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Define test suites
        self.test_suites = {
            "unit": [
                "comprehensive_test_suite.py"
            ],
            "integration": [
                "integration_tests.py"
            ],
            "stress": [
                "stress_testing.py"
            ],
            "scale": [
                "scale_validation.py"
            ]
        }
        
        # Test configuration
        self.config = {
            "coverage": True,
            "xml_output": True,
            "html_report": True,
            "parallel_execution": False,
            "fail_fast": False
        }
    
    def run_python_module(self, module_path: str, output_file: str = None) -> Dict[str, Any]:
        """Run a Python test module and capture output."""
        start_time = time.time()
        
        cmd = [sys.executable, str(module_path)]
        
        try:
            if output_file:
                with open(output_file, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        cwd=self.test_directory.parent,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=300  # 5 minute timeout
                    )
            else:
                result = subprocess.run(
                    cmd,
                    cwd=self.test_directory.parent,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            
            execution_time = time.time() - start_time
            
            return {
                "returncode": result.returncode,
                "execution_time": execution_time,
                "stdout": getattr(result, 'stdout', ''),
                "stderr": getattr(result, 'stderr', ''),
                "success": result.returncode == 0
            }
        
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "returncode": -1,
                "execution_time": execution_time,
                "stdout": "",
                "stderr": "Test timed out after 300 seconds",
                "success": False
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "returncode": -1,
                "execution_time": execution_time,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
    
    def run_unittest_suite(self, test_file: str, xml_output: str = None) -> Dict[str, Any]:
        """Run a unittest-based test suite."""
        start_time = time.time()
        
        # Construct command to run unittest with XML output
        cmd = [sys.executable, "-m", "unittest", f"discover", "-s", str(self.test_directory), "-p", test_file]
        
        if xml_output:
            # Add coverage and XML output options
            cmd = [sys.executable, "-m", "coverage", "run", "--source=src/", "-m", 
                   "unittest", f"discover", "-s", str(self.test_directory), "-p", test_file]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_directory.parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for unittest
            )
            
            execution_time = time.time() - start_time
            
            # If using coverage, generate XML report
            if xml_output and result.returncode == 0:
                cov_cmd = [sys.executable, "-m", "coverage", "xml", "-o", xml_output]
                subprocess.run(
                    cov_cmd,
                    cwd=self.test_directory.parent,
                    capture_output=True,
                    text=True
                )
                
                # Also generate HTML report
                html_dir = xml_output.replace('.xml', '_html')
                cov_html_cmd = [sys.executable, "-m", "coverage", "html", "-d", html_dir]
                subprocess.run(
                    cov_html_cmd,
                    cwd=self.test_directory.parent,
                    capture_output=True,
                    text=True
                )
            
            return {
                "returncode": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "returncode": -1,
                "execution_time": execution_time,
                "stdout": "",
                "stderr": "Unittest timed out after 600 seconds",
                "success": False
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "returncode": -1,
                "execution_time": execution_time,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
    
    def run_test_suite(self, suite_name: str, test_files: List[str]) -> Dict[str, Any]:
        """Run a specific test suite."""
        print(f"Running {suite_name} test suite...")
        
        suite_results = {
            "suite_name": suite_name,
            "start_time": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_execution_time": 0,
            "individual_results": [],
            "success": True
        }
        
        for test_file in test_files:
            test_path = self.test_directory / test_file
            
            if not test_path.exists():
                print(f"Warning: Test file {test_file} does not exist")
                continue
            
            print(f"  Running {test_file}...")
            
            # Determine if it's a unittest file or script
            if "test" in test_file and test_file.endswith(".py"):
                # For comprehensive and integration tests, use unittest discovery
                if "comprehensive" in test_file or "integration" in test_file:
                    xml_output = str(self.reports_dir / f"{suite_name}_{test_file.replace('.py', '.xml')}")
                    result = self.run_unittest_suite(test_file, xml_output)
                else:
                    # For other test files, run as scripts
                    output_file = str(self.reports_dir / f"{suite_name}_{test_file.replace('.py', '.out')}")
                    result = self.run_python_module(test_path, output_file)
            else:
                output_file = str(self.reports_dir / f"{suite_name}_{test_file.replace('.py', '.out')}")
                result = self.run_python_module(test_path, output_file)
            
            # Update suite results
            suite_results["tests_run"] += 1
            if result["success"]:
                suite_results["tests_passed"] += 1
            else:
                suite_results["tests_failed"] += 1
                suite_results["success"] = False
            
            suite_results["total_execution_time"] += result["execution_time"]
            
            # Add individual result
            suite_results["individual_results"].append({
                "test_file": test_file,
                "result": result
            })
            
            # Print status
            status = "PASS" if result["success"] else "FAIL"
            print(f"    {test_file}: {status} ({result['execution_time']:.2f}s)")
            
            # Fail fast if configured
            if self.config["fail_fast"] and not result["success"]:
                print("Fail fast enabled - stopping test suite")
                break
        
        suite_results["end_time"] = datetime.now().isoformat()
        
        return suite_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        print("Starting automated test execution pipeline...")
        print(f"Report directory: {self.reports_dir.absolute()}")
        print()
        
        pipeline_results = {
            "pipeline_start_time": datetime.now().isoformat(),
            "suites_run": 0,
            "suites_passed": 0,
            "suites_failed": 0,
            "total_execution_time": 0,
            "suite_results": {},
            "success": True
        }
        
        for suite_name, test_files in self.test_suites.items():
            print(f"{'='*60}")
            print(f"EXECUTING {suite_name.upper()} TEST SUITE")
            print(f"{'='*60}")
            
            suite_result = self.run_test_suite(suite_name, test_files)
            
            pipeline_results["suites_run"] += 1
            if suite_result["success"]:
                pipeline_results["suites_passed"] += 1
            else:
                pipeline_results["suites_failed"] += 1
                pipeline_results["success"] = False
            
            pipeline_results["total_execution_time"] += suite_result["total_execution_time"]
            pipeline_results["suite_results"][suite_name] = suite_result
            
            print(f"\n{suite_name.upper()} SUITE SUMMARY:")
            print(f"  Tests run: {suite_result['tests_run']}")
            print(f"  Passed: {suite_result['tests_passed']}")
            print(f"  Failed: {suite_result['tests_failed']}")
            print(f"  Execution time: {suite_result['total_execution_time']:.2f}s")
            print(f"  Status: {'PASS' if suite_result['success'] else 'FAIL'}")
            print()
        
        pipeline_results["pipeline_end_time"] = datetime.now().isoformat()
        
        # Generate summary report
        self.generate_summary_report(pipeline_results)
        
        return pipeline_results
    
    def generate_summary_report(self, pipeline_results: Dict[str, Any]):
        """Generate a summary report of the test execution."""
        report_path = self.reports_dir / "test_summary_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Credithos EKM Test Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .suite {{ border: 1px solid #ccc; margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Credithos EKM Test Pipeline Report</h1>
        <p><strong>Start Time:</strong> {pipeline_results['pipeline_start_time']}</p>
        <p><strong>End Time:</strong> {pipeline_results['pipeline_end_time']}</p>
        <p><strong>Total Execution Time:</strong> {pipeline_results['total_execution_time']:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Pipeline Summary</h2>
        <p><strong>Suites Run:</strong> {pipeline_results['suites_run']}</p>
        <p><strong>Suites Passed:</strong> <span class="{'pass' if pipeline_results['success'] else 'fail'}">
            {pipeline_results['suites_passed']}
        </span></p>
        <p><strong>Suites Failed:</strong> <span class="{'fail' if pipeline_results['suites_failed'] > 0 else 'pass'}">
            {pipeline_results['suites_failed']}
        </span></p>
        <p><strong>Overall Status:</strong> <span class="{'pass' if pipeline_results['success'] else 'fail'}">
            {'PASS' if pipeline_results['success'] else 'FAIL'}
        </span></p>
    </div>
    
    <h2>Detailed Results</h2>
"""
        
        for suite_name, suite_result in pipeline_results['suite_results'].items():
            html_content += f"""
    <div class="suite">
        <h3>{suite_name.title()} Test Suite</h3>
        <p><strong>Tests Run:</strong> {suite_result['tests_run']}</p>
        <p><strong>Passed:</strong> {suite_result['tests_passed']}</p>
        <p><strong>Failed:</strong> {suite_result['tests_failed']}</p>
        <p><strong>Execution Time:</strong> {suite_result['total_execution_time']:.2f}s</p>
        <p><strong>Status:</strong> <span class="{'pass' if suite_result['success'] else 'fail'}">
            {'PASS' if suite_result['success'] else 'FAIL'}
        </span></p>
        
        <h4>Individual Test Results:</h4>
        <table>
            <tr><th>Test File</th><th>Status</th><th>Time (s)</th><th>Details</th></tr>
"""
            
            for test_result in suite_result['individual_results']:
                test_file = test_result['test_file']
                result = test_result['result']
                status = '<span class="pass">PASS</span>' if result['success'] else '<span class="fail">FAIL</span>'
                details = f"Return code: {result['returncode']}"
                
                html_content += f"""
            <tr>
                <td>{test_file}</td>
                <td>{status}</td>
                <td>{result['execution_time']:.2f}</td>
                <td>{details}</td>
            </tr>
"""
            
            html_content += """
        </table>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Summary report generated: {report_path.absolute()}")
    
    def generate_json_report(self, pipeline_results: Dict[str, Any]):
        """Generate a JSON report of the test execution."""
        report_path = self.reports_dir / "test_results.json"
        
        with open(report_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"JSON report generated: {report_path.absolute()}")
    
    def run_specific_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        print(f"Running specific suite: {suite_name}")
        suite_result = self.run_test_suite(suite_name, self.test_suites[suite_name])
        
        # Generate suite-specific report
        report_path = self.reports_dir / f"{suite_name}_results.json"
        with open(report_path, 'w') as f:
            json.dump(suite_result, f, indent=2)
        
        print(f"Suite report generated: {report_path.absolute()}")
        
        return suite_result


def main():
    """Main entry point for the test pipeline."""
    parser = argparse.ArgumentParser(description="Credithos EKM Test Pipeline")
    parser.add_argument(
        "--suite", 
        choices=["unit", "integration", "stress", "scale", "all"], 
        default="all",
        help="Specific test suite to run (default: all)"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="test_reports",
        help="Directory for test reports (default: test_reports)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TestPipeline(reports_dir=args.reports_dir)
    
    # Run tests based on argument
    if args.suite == "all":
        results = pipeline.run_all_tests()
    else:
        if args.suite in pipeline.test_suites:
            results = pipeline.run_specific_suite(args.suite)
        else:
            print(f"Unknown suite: {args.suite}")
            return 1
    
    # Generate JSON report
    if args.suite == "all":
        pipeline.generate_json_report(results)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PIPELINE EXECUTION COMPLETE")
    print(f"{'='*60}")
    
    if args.suite == "all":
        print(f"Suites passed: {results['suites_passed']}/{results['suites_run']}")
        print(f"Overall status: {'PASS' if results['success'] else 'FAIL'}")
    else:
        status = results['success']
        print(f"Suite '{args.suite}' status: {'PASS' if status else 'FAIL'}")
    
    print(f"Reports directory: {pipeline.reports_dir.absolute()}")
    
    # Exit with appropriate code
    return 0 if (results['success'] if isinstance(results, dict) else True) else 1


if __name__ == "__main__":
    sys.exit(main())