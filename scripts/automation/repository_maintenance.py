#!/usr/bin/env python3
"""
Repository maintenance automation script for liquid-audio-nets.
Performs routine maintenance tasks like cleanup, optimization, and health checks.
"""

import os
import subprocess
import shutil
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
    return result

def cleanup_build_artifacts():
    """Clean up build artifacts and temporary files."""
    print("🧹 Cleaning build artifacts...")
    
    # Rust build artifacts
    if Path("target").exists():
        print("  Cleaning Rust target directory...")
        run_command("cargo clean", check=False)
    
    # Python build artifacts
    patterns_to_remove = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo", 
        "**/*.egg-info",
        "**/build",
        "**/dist",
        "**/.pytest_cache",
        "**/htmlcov",
        "**/.coverage",
        "coverage.xml"
    ]
    
    for pattern in patterns_to_remove:
        for path in Path(".").glob(pattern):
            if path.exists():
                print(f"  Removing {path}...")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    
    # C++ build artifacts
    if Path("build").exists():
        print("  Cleaning C++ build directory...")
        shutil.rmtree("build")
    
    # Docker build cache (optional)
    docker_cleanup = input("🐳 Clean Docker build cache? [y/N]: ")
    if docker_cleanup.lower() == 'y':
        run_command("docker system prune -f", check=False)
    
    print("✅ Build artifacts cleaned!")

def update_dependencies():
    """Update project dependencies."""
    print("📦 Updating dependencies...")
    
    # Update Python dependencies
    if Path("pyproject.toml").exists():
        print("  Updating Python dependencies...")
        run_command("pip install --upgrade pip", check=False)
        run_command("pip install -e .[dev] --upgrade", check=False)
    
    # Update Rust dependencies
    if Path("Cargo.toml").exists():
        print("  Updating Rust dependencies...")
        run_command("cargo update", check=False)
    
    # Update pre-commit hooks
    if Path(".pre-commit-config.yaml").exists():
        print("  Updating pre-commit hooks...")
        run_command("pre-commit autoupdate", check=False)
    
    print("✅ Dependencies updated!")

def run_security_checks():
    """Run security checks and audits."""
    print("🔒 Running security checks...")
    
    issues_found = []
    
    # Python security check
    result = run_command("pip install safety", check=False)
    if result.returncode == 0:
        result = run_command("safety check", check=False)
        if result.returncode != 0:
            issues_found.append("Python security vulnerabilities detected")
    
    # Rust security audit
    result = run_command("cargo install cargo-audit", check=False)
    if result.returncode == 0:
        result = run_command("cargo audit", check=False)
        if result.returncode != 0:
            issues_found.append("Rust security vulnerabilities detected")
    
    # Secret detection
    result = run_command("pip install detect-secrets", check=False)
    if result.returncode == 0:
        result = run_command("detect-secrets scan --all-files --baseline .secrets.baseline", check=False)
        if result.returncode != 0:
            issues_found.append("Potential secrets detected")
    
    if issues_found:
        print("⚠️  Security issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No security issues found!")
        return True

def optimize_repository():
    """Optimize repository structure and performance."""
    print("⚡ Optimizing repository...")
    
    # Git maintenance
    print("  Running git maintenance...")
    run_command("git gc --prune=now", check=False)
    run_command("git remote prune origin", check=False)
    
    # Optimize images (if any)
    image_files = list(Path(".").glob("**/*.png")) + list(Path(".").glob("**/*.jpg"))
    if image_files:
        print(f"  Found {len(image_files)} image files")
        optimize_images = input("🖼️  Optimize images? [y/N]: ")
        if optimize_images.lower() == 'y':
            try:
                run_command("which optipng", check=False)
                for img in Path(".").glob("**/*.png"):
                    run_command(f"optipng {img}", check=False)
            except:
                print("  optipng not available, skipping PNG optimization")
    
    # Update file permissions
    print("  Updating file permissions...")
    for script in Path("scripts").glob("**/*.py"):
        if script.is_file():
            script.chmod(0o755)
    
    for script in Path("scripts").glob("**/*.sh"):
        if script.is_file():
            script.chmod(0o755)
    
    print("✅ Repository optimized!")

def check_repository_health() -> Dict[str, Any]:
    """Check overall repository health."""
    print("🏥 Checking repository health...")
    
    health_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checks": {},
        "overall_status": "healthy"
    }
    
    # Check if core files exist
    core_files = [
        "README.md", "LICENSE", "pyproject.toml", "Cargo.toml", 
        "CMakeLists.txt", "Makefile", ".gitignore", ".pre-commit-config.yaml"
    ]
    
    missing_files = []
    for file in core_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    health_report["checks"]["core_files"] = {
        "status": "pass" if not missing_files else "fail",
        "missing_files": missing_files
    }
    
    # Check build system
    rust_builds = run_command("cargo check", check=False).returncode == 0
    python_imports = run_command("python -c 'import liquid_audio_nets'", check=False).returncode == 0
    
    health_report["checks"]["build_system"] = {
        "rust_builds": rust_builds,
        "python_imports": python_imports,
        "status": "pass" if rust_builds and python_imports else "fail"
    }
    
    # Check test suite
    test_result = run_command("python -m pytest tests/ --collect-only -q", check=False)
    test_count = len([line for line in test_result.stdout.split('\n') if 'test' in line])
    
    health_report["checks"]["tests"] = {
        "test_count": test_count,
        "status": "pass" if test_count > 0 else "warn"
    }
    
    # Check documentation
    doc_files = list(Path("docs").glob("**/*.md")) if Path("docs").exists() else []
    
    health_report["checks"]["documentation"] = {
        "doc_count": len(doc_files),
        "status": "pass" if len(doc_files) > 5 else "warn"
    }
    
    # Check for large files
    large_files = []
    for file in Path(".").rglob("*"):
        if file.is_file() and file.stat().st_size > 10 * 1024 * 1024:  # 10MB
            large_files.append(str(file))
    
    health_report["checks"]["large_files"] = {
        "count": len(large_files),
        "files": large_files,
        "status": "warn" if large_files else "pass"
    }
    
    # Determine overall status
    failed_checks = [check for check in health_report["checks"].values() if check["status"] == "fail"]
    warn_checks = [check for check in health_report["checks"].values() if check["status"] == "warn"]
    
    if failed_checks:
        health_report["overall_status"] = "unhealthy"
    elif warn_checks:
        health_report["overall_status"] = "warning"
    
    # Save health report
    with open("health-report.json", "w") as f:
        json.dump(health_report, f, indent=2)
    
    print("✅ Health check complete!")
    return health_report

def generate_maintenance_report(health_report: Dict[str, Any]):
    """Generate a maintenance report."""
    print("📊 Generating maintenance report...")
    
    report_lines = [
        "# Repository Maintenance Report",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Overall Status:** {health_report['overall_status'].upper()}",
        "",
        "## Health Checks",
        ""
    ]
    
    for check_name, check_data in health_report["checks"].items():
        status_emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[check_data["status"]]
        report_lines.append(f"### {check_name.replace('_', ' ').title()}")
        report_lines.append(f"**Status:** {status_emoji} {check_data['status'].upper()}")
        
        if check_name == "core_files" and check_data.get("missing_files"):
            report_lines.append(f"**Missing files:** {', '.join(check_data['missing_files'])}")
        elif check_name == "build_system":
            report_lines.append(f"**Rust builds:** {'✅' if check_data['rust_builds'] else '❌'}")
            report_lines.append(f"**Python imports:** {'✅' if check_data['python_imports'] else '❌'}")
        elif check_name == "tests":
            report_lines.append(f"**Test count:** {check_data['test_count']}")
        elif check_name == "documentation":
            report_lines.append(f"**Documentation files:** {check_data['doc_count']}")
        elif check_name == "large_files" and check_data.get("files"):
            report_lines.append(f"**Large files found:** {check_data['count']}")
            for file in check_data["files"][:5]:  # Show first 5
                report_lines.append(f"  - {file}")
        
        report_lines.append("")
    
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    if health_report["overall_status"] == "unhealthy":
        report_lines.append("⚠️ **Immediate action required:** Critical issues detected")
    elif health_report["overall_status"] == "warning":
        report_lines.append("📋 **Maintenance recommended:** Some issues detected")
    else:
        report_lines.append("✅ **Repository is healthy:** No critical issues detected")
    
    report_lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Review and address any failed checks",
        "2. Consider optimizing large files",
        "3. Update dependencies regularly",
        "4. Run security audits periodically",
        "5. Monitor repository metrics",
        ""
    ])
    
    report_content = "\n".join(report_lines)
    
    with open("maintenance-report.md", "w") as f:
        f.write(report_content)
    
    print("📄 Maintenance report saved to maintenance-report.md")
    return report_content

def main():
    """Main maintenance function."""
    print("🔧 Starting repository maintenance...")
    
    # Ask user what to do
    print("\nMaintenance options:")
    print("1. Full maintenance (cleanup + update + security + optimize)")
    print("2. Quick cleanup only")
    print("3. Update dependencies only")
    print("4. Security check only") 
    print("5. Health check only")
    print("6. Custom selection")
    
    choice = input("\nSelect option [1-6]: ").strip()
    
    if choice == "1":
        cleanup_build_artifacts()
        update_dependencies()
        security_ok = run_security_checks()
        optimize_repository()
        health_report = check_repository_health()
        generate_maintenance_report(health_report)
    elif choice == "2":
        cleanup_build_artifacts()
    elif choice == "3":
        update_dependencies()
    elif choice == "4":
        run_security_checks()
    elif choice == "5":
        health_report = check_repository_health()
        generate_maintenance_report(health_report)
    elif choice == "6":
        tasks = []
        if input("Cleanup build artifacts? [y/N]: ").lower() == 'y':
            tasks.append(cleanup_build_artifacts)
        if input("Update dependencies? [y/N]: ").lower() == 'y':
            tasks.append(update_dependencies)
        if input("Run security checks? [y/N]: ").lower() == 'y':
            tasks.append(run_security_checks)
        if input("Optimize repository? [y/N]: ").lower() == 'y':
            tasks.append(optimize_repository)
        if input("Health check? [y/N]: ").lower() == 'y':
            tasks.append(lambda: check_repository_health())
        
        for task in tasks:
            task()
        
        if tasks and any('health' in str(task) for task in tasks):
            health_report = check_repository_health()
            generate_maintenance_report(health_report)
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n✅ Repository maintenance complete!")

if __name__ == "__main__":
    main()