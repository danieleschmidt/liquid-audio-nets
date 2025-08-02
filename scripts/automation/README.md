# Automation Scripts

This directory contains automation scripts for the liquid-audio-nets project that help with maintenance, monitoring, and quality assurance.

## 📊 collect_metrics.py

Automated metrics collection for project health tracking.

### Features
- **Development Metrics**: Lines of code, test coverage, complexity
- **Quality Metrics**: Security issues, linting results, technical debt
- **Performance Metrics**: Benchmark results, regression tracking  
- **Community Metrics**: GitHub stars, issues, PRs, contributors
- **Documentation Metrics**: Coverage and freshness

### Usage
```bash
# Collect all metrics and generate report
python scripts/automation/collect_metrics.py

# Generate report from existing metrics
python scripts/automation/collect_metrics.py --report-only
```

### Environment Variables
- `GITHUB_TOKEN`: For GitHub API access
- `GITHUB_REPOSITORY`: Repository name (auto-detected in CI)

### Outputs
- `.github/project-metrics.json`: Updated metrics data
- `metrics-report.md`: Human-readable report

## 🔧 repository_maintenance.py

Comprehensive repository maintenance and optimization.

### Features
- **Build Cleanup**: Remove build artifacts and temporary files
- **Dependency Updates**: Update Python, Rust, and other dependencies
- **Security Checks**: Run security audits and vulnerability scans
- **Repository Optimization**: Git maintenance, image optimization
- **Health Checks**: Validate repository structure and functionality

### Usage
```bash
# Interactive maintenance menu
python scripts/automation/repository_maintenance.py

# Choose from:
# 1. Full maintenance (all tasks)
# 2. Quick cleanup only
# 3. Update dependencies only
# 4. Security check only
# 5. Health check only
# 6. Custom selection
```

### Outputs
- `health-report.json`: Repository health data
- `maintenance-report.md`: Maintenance summary

## 📦 dependency_manager.py

Advanced dependency management and monitoring.

### Features
- **Dependency Checking**: Check for outdated packages and vulnerabilities
- **Automated Updates**: Update Python and Rust dependencies
- **Security Monitoring**: Track and alert on vulnerabilities
- **Report Generation**: Comprehensive dependency reports
- **PR Automation**: Create automated update pull requests

### Usage
```bash
# Check dependency status
python scripts/automation/dependency_manager.py check

# Update dependencies interactively
python scripts/automation/dependency_manager.py update

# Generate dependency report
python scripts/automation/dependency_manager.py report

# Monitor for critical issues
python scripts/automation/dependency_manager.py monitor

# Create automated update PR
python scripts/automation/dependency_manager.py auto-pr
```

### Outputs
- `dependency-report.json`: Comprehensive dependency analysis
- `dependency-monitoring.json`: Monitoring data and alerts

## 🔄 Integration with CI/CD

### GitHub Actions Integration

Add to your workflow:

```yaml
- name: Collect Metrics
  run: python scripts/automation/collect_metrics.py

- name: Repository Health Check
  run: python scripts/automation/repository_maintenance.py

- name: Dependency Monitoring
  run: python scripts/automation/dependency_manager.py monitor
```

### Scheduled Runs

```yaml
# Weekly maintenance
- cron: '0 8 * * 1'  # Mondays at 8 AM
  run: |
    python scripts/automation/repository_maintenance.py
    python scripts/automation/dependency_manager.py auto-pr

# Daily monitoring  
- cron: '0 6 * * *'  # Daily at 6 AM
  run: |
    python scripts/automation/collect_metrics.py
    python scripts/automation/dependency_manager.py monitor
```

## 🚨 Alerting and Notifications

### Metrics Thresholds

Configure alerts in `.github/project-metrics.json`:

```json
{
  "tracking": {
    "alert_thresholds": {
      "test_coverage_drop": 5,
      "security_issues_increase": 1,
      "performance_regression": 10,
      "build_failure_rate": 10
    }
  }
}
```

### Slack Integration

Set up Slack webhooks for notifications:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
```

### Email Alerts

Configure email notifications in your CI/CD system when:
- Security vulnerabilities are detected
- Critical health checks fail
- Performance regressions occur
- Dependency issues arise

## 📋 Maintenance Schedule

### Daily
- Metrics collection
- Dependency monitoring
- Security vulnerability checks

### Weekly
- Full repository maintenance
- Dependency updates (automated PR)
- Performance benchmark analysis

### Monthly
- Comprehensive health audit
- Technical debt assessment
- Documentation review

## 🔧 Configuration

### Dependencies

Install required Python packages:

```bash
pip install requests toml safety bandit semgrep
```

Install required tools:

```bash
# Rust tools
cargo install cargo-audit cargo-outdated cargo-deny

# Python tools
pip install detect-secrets pre-commit

# Optional tools
apt-get install optipng  # For image optimization
```

### GitHub CLI (for PR automation)

```bash
# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh

# Authenticate
gh auth login
```

## 🔍 Troubleshooting

### Common Issues

**Metrics collection fails:**
- Check GitHub token permissions
- Verify repository access
- Ensure required tools are installed

**Repository maintenance errors:**
- Check file permissions
- Verify disk space
- Ensure git repository is clean

**Dependency management issues:**
- Check internet connectivity
- Verify package registry access
- Ensure proper authentication

### Debug Mode

Run scripts with debug output:

```bash
export DEBUG=1
python scripts/automation/collect_metrics.py
```

### Log Files

Check log files for detailed error information:
- `automation.log`: General automation logs
- `metrics.log`: Metrics collection logs
- `maintenance.log`: Maintenance operation logs

## 🚀 Best Practices

### Security
- Store sensitive tokens in environment variables
- Use least-privilege access for automation
- Regularly rotate tokens and credentials
- Audit automation script permissions

### Performance
- Run heavy operations during low-traffic hours
- Use caching for expensive operations
- Implement rate limiting for API calls
- Monitor automation resource usage

### Reliability
- Implement retry logic for network operations
- Use proper error handling and logging
- Test automation scripts in staging environment
- Have fallback procedures for critical automation

### Monitoring
- Track automation script execution times
- Monitor success/failure rates
- Set up alerts for automation failures
- Regularly review and update automation logic

## 📞 Support

For automation-related issues:

1. Check script logs and error messages
2. Review the troubleshooting section
3. Verify configuration and dependencies
4. Open an issue with the `automation` label
5. Contact the DevOps team for assistance

## 🔗 Related Documentation

- [GitHub Actions Workflows](../workflows/README.md)
- [Monitoring Setup](../../monitoring/README.md)
- [Development Guidelines](../../docs/DEVELOPMENT.md)
- [Security Policies](../../SECURITY.md)