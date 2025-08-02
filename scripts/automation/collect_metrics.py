#!/usr/bin/env python3
"""
Automated metrics collection script for liquid-audio-nets project.
Collects and updates project metrics in .github/project-metrics.json
"""

import json
import subprocess
import datetime
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import requests

def run_command(cmd: str, cwd: Optional[str] = None) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd.split(), 
            capture_output=True, 
            text=True, 
            cwd=cwd
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")
        return ""

def count_lines_of_code(directory: str, extensions: list) -> int:
    """Count lines of code for given file extensions."""
    total_lines = 0
    for ext in extensions:
        try:
            result = run_command(f"find {directory} -name '*.{ext}' -exec wc -l {{}} +")
            if result:
                lines = result.split('\n')[-2].split()[0] if lines else "0"
                total_lines += int(lines)
        except:
            pass
    return total_lines

def get_test_coverage() -> float:
    """Get test coverage percentage."""
    try:
        # Run pytest with coverage
        result = run_command("python -m pytest --cov=liquid_audio_nets --cov-report=json")
        if os.path.exists("coverage.json"):
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)
                return coverage_data.get("totals", {}).get("percent_covered", 0)
    except:
        pass
    return 0

def get_security_issues() -> Dict[str, int]:
    """Get security issues count."""
    issues = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    try:
        # Run safety check for Python
        result = run_command("safety check --json")
        if result:
            safety_data = json.loads(result)
            for vuln in safety_data:
                severity = vuln.get("severity", "low").lower()
                if severity in issues:
                    issues[severity] += 1
    except:
        pass
    
    try:
        # Run cargo audit for Rust
        result = run_command("cargo audit --json")
        if result:
            audit_data = json.loads(result)
            vulnerabilities = audit_data.get("vulnerabilities", {}).get("list", [])
            for vuln in vulnerabilities:
                # Cargo audit doesn't provide severity, assume medium
                issues["medium"] += 1
    except:
        pass
    
    return issues

def get_github_metrics(repo: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Get GitHub repository metrics."""
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        # Get repository info
        response = requests.get(f"https://api.github.com/repos/{repo}", headers=headers)
        if response.status_code == 200:
            repo_data = response.json()
            
            # Get issues and PRs
            issues_response = requests.get(f"https://api.github.com/repos/{repo}/issues?state=open", headers=headers)
            prs_response = requests.get(f"https://api.github.com/repos/{repo}/pulls?state=open", headers=headers)
            
            open_issues = len(issues_response.json()) if issues_response.status_code == 200 else 0
            open_prs = len(prs_response.json()) if prs_response.status_code == 200 else 0
            
            return {
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "open_issues": open_issues,
                "open_prs": open_prs,
                "contributors": repo_data.get("subscribers_count", 0)
            }
    except Exception as e:
        print(f"Error fetching GitHub metrics: {e}")
    
    return {"stars": 0, "forks": 0, "open_issues": 0, "open_prs": 0, "contributors": 0}

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance benchmark results."""
    metrics = {
        "inference_latency_ms": {"current": None, "trend": "stable"},
        "power_consumption_mw": {"current": None, "trend": "stable"},
        "memory_usage_kb": {"current": None, "trend": "stable"},
        "accuracy_percent": {"current": None, "trend": "stable"}
    }
    
    try:
        # Check if benchmark results exist
        benchmark_file = "benchmark-results.json"
        if os.path.exists(benchmark_file):
            with open(benchmark_file, "r") as f:
                benchmark_data = json.load(f)
                
                for benchmark in benchmark_data.get("benchmarks", []):
                    name = benchmark.get("name", "")
                    if "latency" in name.lower():
                        metrics["inference_latency_ms"]["current"] = benchmark.get("stats", {}).get("mean", None)
                    elif "power" in name.lower():
                        metrics["power_consumption_mw"]["current"] = benchmark.get("stats", {}).get("mean", None)
                    elif "memory" in name.lower():
                        metrics["memory_usage_kb"]["current"] = benchmark.get("stats", {}).get("mean", None)
                    elif "accuracy" in name.lower():
                        metrics["accuracy_percent"]["current"] = benchmark.get("stats", {}).get("mean", None)
    except:
        pass
    
    return metrics

def collect_all_metrics() -> Dict[str, Any]:
    """Collect all project metrics."""
    print("🔍 Collecting project metrics...")
    
    # Load existing metrics
    metrics_file = Path(".github/project-metrics.json")
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    else:
        print("❌ Metrics file not found")
        return {}
    
    # Update timestamp
    metrics["last_updated"] = datetime.datetime.now().isoformat()
    
    # Collect development metrics
    print("📊 Collecting development metrics...")
    rust_loc = count_lines_of_code("src", ["rs"])
    python_loc = count_lines_of_code("python", ["py"])
    cpp_loc = count_lines_of_code("cpp", ["cpp", "c", "h", "hpp"])
    
    metrics["metrics"]["development"]["languages"]["rust"]["lines_of_code"] = rust_loc
    metrics["metrics"]["development"]["languages"]["python"]["lines_of_code"] = python_loc
    metrics["metrics"]["development"]["languages"]["cpp"]["lines_of_code"] = cpp_loc
    metrics["metrics"]["development"]["total_lines_of_code"] = rust_loc + python_loc + cpp_loc
    
    # Collect test coverage
    print("🧪 Collecting test coverage...")
    coverage = get_test_coverage()
    metrics["metrics"]["development"]["total_test_coverage"] = coverage
    metrics["metrics"]["testing"]["unit_tests"]["coverage"] = coverage
    
    # Collect security metrics
    print("🔒 Collecting security metrics...")
    security_issues = get_security_issues()
    metrics["metrics"]["quality"]["security_issues"] = security_issues
    metrics["metrics"]["quality"]["security_issues"]["last_scan"] = datetime.datetime.now().isoformat()
    
    # Collect GitHub metrics
    print("🐙 Collecting GitHub metrics...")
    repo = os.environ.get("GITHUB_REPOSITORY", "danieleschmidt/liquid-audio-nets")
    github_token = os.environ.get("GITHUB_TOKEN")
    github_metrics = get_github_metrics(repo, github_token)
    
    metrics["metrics"]["community"]["stars"] = github_metrics["stars"]
    metrics["metrics"]["community"]["forks"] = github_metrics["forks"]
    metrics["metrics"]["community"]["issues"]["open"] = github_metrics["open_issues"]
    metrics["metrics"]["community"]["pull_requests"]["open"] = github_metrics["open_prs"]
    metrics["metrics"]["community"]["contributors"] = github_metrics["contributors"]
    
    # Collect performance metrics
    print("⚡ Collecting performance metrics...")
    performance_metrics = get_performance_metrics()
    metrics["metrics"]["performance"]["benchmarks"] = performance_metrics
    
    # Count test files
    test_count = len(list(Path("tests").rglob("test_*.py"))) if Path("tests").exists() else 0
    metrics["metrics"]["testing"]["unit_tests"]["total"] = test_count
    
    # Count documentation files
    doc_count = len(list(Path("docs").rglob("*.md"))) if Path("docs").exists() else 0
    metrics["metrics"]["documentation"]["pages_count"] = doc_count
    
    print("✅ Metrics collection complete!")
    return metrics

def generate_report(metrics: Dict[str, Any]) -> str:
    """Generate a metrics report."""
    report = ["# Project Metrics Report", ""]
    report.append(f"**Generated:** {metrics.get('last_updated', 'Unknown')}")
    report.append("")
    
    # Development metrics
    dev_metrics = metrics.get("metrics", {}).get("development", {})
    report.append("## 📊 Development Metrics")
    report.append(f"- **Total Lines of Code:** {dev_metrics.get('total_lines_of_code', 0):,}")
    report.append(f"- **Test Coverage:** {dev_metrics.get('total_test_coverage', 0):.1f}%")
    report.append("")
    
    # Quality metrics
    quality_metrics = metrics.get("metrics", {}).get("quality", {})
    security_issues = quality_metrics.get("security_issues", {})
    total_security = sum(security_issues.get(k, 0) for k in ["critical", "high", "medium", "low"])
    report.append("## 🔒 Quality Metrics")
    report.append(f"- **Security Issues:** {total_security}")
    report.append(f"  - Critical: {security_issues.get('critical', 0)}")
    report.append(f"  - High: {security_issues.get('high', 0)}")
    report.append(f"  - Medium: {security_issues.get('medium', 0)}")
    report.append(f"  - Low: {security_issues.get('low', 0)}")
    report.append("")
    
    # Community metrics
    community_metrics = metrics.get("metrics", {}).get("community", {})
    report.append("## 👥 Community Metrics")
    report.append(f"- **Stars:** {community_metrics.get('stars', 0)}")
    report.append(f"- **Forks:** {community_metrics.get('forks', 0)}")
    report.append(f"- **Open Issues:** {community_metrics.get('issues', {}).get('open', 0)}")
    report.append(f"- **Open PRs:** {community_metrics.get('pull_requests', {}).get('open', 0)}")
    report.append("")
    
    # Performance metrics
    performance_metrics = metrics.get("metrics", {}).get("performance", {}).get("benchmarks", {})
    report.append("## ⚡ Performance Metrics")
    for metric, data in performance_metrics.items():
        current = data.get("current")
        if current is not None:
            report.append(f"- **{metric.replace('_', ' ').title()}:** {current}")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--report-only":
        # Generate report from existing metrics
        metrics_file = Path(".github/project-metrics.json")
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            report = generate_report(metrics)
            print(report)
        else:
            print("❌ No metrics file found")
        return
    
    # Collect metrics
    metrics = collect_all_metrics()
    if not metrics:
        sys.exit(1)
    
    # Save metrics
    metrics_file = Path(".github/project-metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📄 Metrics saved to {metrics_file}")
    
    # Generate and display report
    report = generate_report(metrics)
    print("\n" + report)
    
    # Save report
    report_file = Path("metrics-report.md")
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"📊 Report saved to {report_file}")

if __name__ == "__main__":
    main()