#!/usr/bin/env python3
"""
Dependency management automation for liquid-audio-nets.
Handles dependency updates, security monitoring, and compatibility checking.
"""

import json
import subprocess
import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import toml

def run_command(cmd: str, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd.split() if isinstance(cmd, str) else cmd,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_python_dependencies() -> Dict[str, any]:
    """Check Python dependencies for updates and vulnerabilities."""
    print("🐍 Checking Python dependencies...")
    
    result = {
        "outdated": [],
        "vulnerabilities": [],
        "total_packages": 0
    }
    
    # Get installed packages
    exit_code, stdout, stderr = run_command("pip list --format=json")
    if exit_code == 0:
        packages = json.loads(stdout)
        result["total_packages"] = len(packages)
        
        # Check for outdated packages
        exit_code, stdout, stderr = run_command("pip list --outdated --format=json")
        if exit_code == 0:
            outdated = json.loads(stdout)
            result["outdated"] = outdated
    
    # Check for vulnerabilities with safety
    exit_code, stdout, stderr = run_command("safety check --json")
    if exit_code != 0 and stdout:
        try:
            vulnerabilities = json.loads(stdout)
            result["vulnerabilities"] = vulnerabilities
        except:
            pass
    
    return result

def check_rust_dependencies() -> Dict[str, any]:
    """Check Rust dependencies for updates and vulnerabilities."""
    print("🦀 Checking Rust dependencies...")
    
    result = {
        "outdated": [],
        "vulnerabilities": [],
        "total_crates": 0
    }
    
    if not Path("Cargo.toml").exists():
        return result
    
    # Parse Cargo.toml
    try:
        with open("Cargo.toml", "r") as f:
            cargo_data = toml.load(f)
        
        dependencies = cargo_data.get("dependencies", {})
        dev_dependencies = cargo_data.get("dev-dependencies", {})
        result["total_crates"] = len(dependencies) + len(dev_dependencies)
    except:
        pass
    
    # Check for outdated crates
    exit_code, stdout, stderr = run_command("cargo outdated --format json")
    if exit_code == 0:
        try:
            outdated_data = json.loads(stdout)
            result["outdated"] = outdated_data.get("dependencies", [])
        except:
            pass
    
    # Check for vulnerabilities
    exit_code, stdout, stderr = run_command("cargo audit --json")
    if exit_code != 0 and stdout:
        try:
            audit_data = json.loads(stdout)
            vulnerabilities = audit_data.get("vulnerabilities", {}).get("list", [])
            result["vulnerabilities"] = vulnerabilities
        except:
            pass
    
    return result

def update_python_dependencies(interactive: bool = True) -> bool:
    """Update Python dependencies."""
    print("📦 Updating Python dependencies...")
    
    if interactive:
        confirm = input("Update all Python packages? [y/N]: ")
        if confirm.lower() != 'y':
            return False
    
    # Update pip first
    exit_code, stdout, stderr = run_command("pip install --upgrade pip")
    if exit_code != 0:
        print(f"Failed to update pip: {stderr}")
        return False
    
    # Update packages
    exit_code, stdout, stderr = run_command("pip install -e .[dev] --upgrade")
    if exit_code != 0:
        print(f"Failed to update packages: {stderr}")
        return False
    
    print("✅ Python dependencies updated!")
    return True

def update_rust_dependencies(interactive: bool = True) -> bool:
    """Update Rust dependencies."""
    print("🔄 Updating Rust dependencies...")
    
    if not Path("Cargo.toml").exists():
        print("No Cargo.toml found, skipping Rust updates")
        return True
    
    if interactive:
        confirm = input("Update all Rust crates? [y/N]: ")
        if confirm.lower() != 'y':
            return False
    
    # Update dependencies
    exit_code, stdout, stderr = run_command("cargo update")
    if exit_code != 0:
        print(f"Failed to update Rust dependencies: {stderr}")
        return False
    
    # Check if everything still builds
    print("🔨 Verifying build after update...")
    exit_code, stdout, stderr = run_command("cargo check")
    if exit_code != 0:
        print(f"Build failed after update: {stderr}")
        return False
    
    print("✅ Rust dependencies updated!")
    return True

def generate_dependency_report() -> Dict[str, any]:
    """Generate comprehensive dependency report."""
    print("📊 Generating dependency report...")
    
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python": check_python_dependencies(),
        "rust": check_rust_dependencies(),
        "summary": {}
    }
    
    # Calculate summary
    total_outdated = len(report["python"]["outdated"]) + len(report["rust"]["outdated"])
    total_vulnerabilities = len(report["python"]["vulnerabilities"]) + len(report["rust"]["vulnerabilities"])
    total_packages = report["python"]["total_packages"] + report["rust"]["total_crates"]
    
    report["summary"] = {
        "total_packages": total_packages,
        "total_outdated": total_outdated,
        "total_vulnerabilities": total_vulnerabilities,
        "health_status": "critical" if total_vulnerabilities > 0 else "warning" if total_outdated > 10 else "good"
    }
    
    return report

def create_update_pull_request(report: Dict[str, any]) -> bool:
    """Create a pull request with dependency updates."""
    print("🔀 Creating dependency update PR...")
    
    if not os.environ.get("GITHUB_TOKEN"):
        print("No GitHub token found, cannot create PR")
        return False
    
    # Create branch
    branch_name = f"auto-update-deps-{datetime.datetime.now().strftime('%Y%m%d')}"
    exit_code, stdout, stderr = run_command(f"git checkout -b {branch_name}")
    if exit_code != 0:
        print(f"Failed to create branch: {stderr}")
        return False
    
    # Update dependencies
    python_updated = update_python_dependencies(interactive=False)
    rust_updated = update_rust_dependencies(interactive=False)
    
    if not python_updated and not rust_updated:
        print("No dependencies were updated")
        return False
    
    # Commit changes
    exit_code, stdout, stderr = run_command("git add .")
    if exit_code != 0:
        print(f"Failed to stage changes: {stderr}")
        return False
    
    commit_msg = "chore(deps): automated dependency updates"
    exit_code, stdout, stderr = run_command(f"git commit -m '{commit_msg}'")
    if exit_code != 0:
        print("No changes to commit")
        return False
    
    # Push branch
    exit_code, stdout, stderr = run_command(f"git push origin {branch_name}")
    if exit_code != 0:
        print(f"Failed to push branch: {stderr}")
        return False
    
    # Create PR (requires gh CLI)
    pr_body = f"""
## Automated Dependency Updates

This PR contains automated dependency updates.

### Summary
- **Total packages:** {report['summary']['total_packages']}
- **Outdated packages:** {report['summary']['total_outdated']}
- **Security vulnerabilities:** {report['summary']['total_vulnerabilities']}

### Python Updates
- **Outdated:** {len(report['python']['outdated'])}
- **Vulnerabilities:** {len(report['python']['vulnerabilities'])}

### Rust Updates  
- **Outdated:** {len(report['rust']['outdated'])}
- **Vulnerabilities:** {len(report['rust']['vulnerabilities'])}

### Testing Required
- [ ] All tests pass
- [ ] No breaking changes
- [ ] Security issues resolved

Auto-generated by dependency manager.
"""
    
    exit_code, stdout, stderr = run_command(f"gh pr create --title 'chore(deps): automated dependency updates' --body '{pr_body}'")
    if exit_code != 0:
        print(f"Failed to create PR: {stderr}")
        return False
    
    print("✅ Pull request created successfully!")
    return True

def monitor_dependencies() -> Dict[str, any]:
    """Monitor dependencies and alert on issues."""
    print("👀 Monitoring dependencies...")
    
    report = generate_dependency_report()
    
    # Check for critical issues
    critical_issues = []
    
    # Check for high-severity vulnerabilities
    for vuln in report["python"]["vulnerabilities"]:
        if vuln.get("severity", "").lower() in ["high", "critical"]:
            critical_issues.append(f"Python: {vuln.get('package')} - {vuln.get('vulnerability')}")
    
    for vuln in report["rust"]["vulnerabilities"]:
        if vuln.get("severity", "").lower() in ["high", "critical"]:
            critical_issues.append(f"Rust: {vuln.get('package')} - {vuln.get('title')}")
    
    # Check for too many outdated packages
    if report["summary"]["total_outdated"] > 20:
        critical_issues.append(f"Too many outdated packages: {report['summary']['total_outdated']}")
    
    # Save monitoring report
    monitoring_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "critical" if critical_issues else "ok",
        "critical_issues": critical_issues,
        "report": report
    }
    
    with open("dependency-monitoring.json", "w") as f:
        json.dump(monitoring_data, f, indent=2)
    
    if critical_issues:
        print("🚨 Critical dependency issues found:")
        for issue in critical_issues:
            print(f"  - {issue}")
    else:
        print("✅ No critical dependency issues found")
    
    return monitoring_data

def main():
    """Main dependency management function."""
    if len(sys.argv) < 2:
        print("Usage: dependency_manager.py <command>")
        print("Commands:")
        print("  check     - Check dependency status")
        print("  update    - Update dependencies interactively")
        print("  report    - Generate dependency report")
        print("  monitor   - Monitor for critical issues")
        print("  auto-pr   - Create automated update PR")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "check":
        python_deps = check_python_dependencies()
        rust_deps = check_rust_dependencies()
        
        print(f"\n📊 Dependency Summary:")
        print(f"Python: {python_deps['total_packages']} packages, {len(python_deps['outdated'])} outdated, {len(python_deps['vulnerabilities'])} vulnerable")
        print(f"Rust: {rust_deps['total_crates']} crates, {len(rust_deps['outdated'])} outdated, {len(rust_deps['vulnerabilities'])} vulnerable")
    
    elif command == "update":
        update_python_dependencies()
        update_rust_dependencies()
    
    elif command == "report":
        report = generate_dependency_report()
        with open("dependency-report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("📄 Report saved to dependency-report.json")
    
    elif command == "monitor":
        monitor_dependencies()
    
    elif command == "auto-pr":
        report = generate_dependency_report()
        create_update_pull_request(report)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()