# GitHub Workflows Setup Guide

This guide provides instructions for setting up GitHub Actions workflows for the liquid-audio-nets project.

## 🚨 Manual Setup Required

Due to GitHub App permission limitations, the workflow files must be created manually by repository administrators.

## 📁 Required Workflow Files

Copy the following files from `docs/workflows/examples/` to `.github/workflows/`:

### Core Workflows

1. **ci.yml** - Continuous Integration
   - Runs tests across multiple Python versions and platforms
   - Performs Rust and C++ builds
   - Executes security scans and linting
   - Validates embedded builds

2. **cd.yml** - Continuous Deployment
   - Builds and publishes packages on releases
   - Creates GitHub releases with artifacts
   - Deploys documentation
   - Updates container registries

3. **security-scan.yml** - Security Scanning
   - Daily security vulnerability scans
   - Secret detection and validation
   - Container security analysis
   - Supply chain security (SBOM generation)

4. **dependency-update.yml** - Automated Dependency Management
   - Weekly dependency updates
   - Security patch automation
   - Compatibility testing
   - Automated PR creation

## 🔧 Setup Instructions

### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Files

```bash
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### Step 3: Configure Repository Secrets

Add the following secrets in GitHub repository settings:

#### Required Secrets
- `PYPI_API_TOKEN` - Token for publishing Python packages to PyPI
- `CRATES_IO_TOKEN` - Token for publishing Rust crates to crates.io

#### Optional Secrets (for enhanced functionality)
- `SLACK_WEBHOOK_URL` - Slack webhook for notifications
- `CODECOV_TOKEN` - Codecov token for coverage reporting
- `DOCKER_HUB_USERNAME` - Docker Hub username
- `DOCKER_HUB_TOKEN` - Docker Hub access token

### Step 4: Configure Branch Protection

Set up branch protection rules for the `main` branch:

```yaml
Required status checks:
  - test-python (3.8, 3.9, 3.10, 3.11)
  - test-rust
  - test-cpp
  - security-scan
  - integration-test

Additional settings:
  - Require branches to be up to date before merging
  - Require linear history
  - Include administrators in restrictions
  - Allow force pushes: false
  - Allow deletions: false
```

### Step 5: Enable Actions Permissions

Configure Actions permissions in repository settings:

- **Actions permissions**: Allow all actions and reusable workflows
- **Workflow permissions**: Read and write permissions
- **Fork pull request workflows**: Require approval for first-time contributors

## 🎯 Workflow Details

### CI Workflow (ci.yml)

**Triggers:**
- Push to main/develop branches
- Pull requests to main branch

**Jobs:**
- `test-python`: Tests across Python 3.8-3.11
- `test-rust`: Rust compilation and testing
- `test-cpp`: C++ compilation and testing
- `security-scan`: Security vulnerability checks
- `performance-test`: Performance benchmarking
- `embedded-test`: Embedded target compilation
- `integration-test`: End-to-end testing
- `documentation`: Documentation building and link checking

**Artifacts:**
- Test coverage reports
- Performance benchmark results
- Security scan reports
- Documentation builds

### CD Workflow (cd.yml)

**Triggers:**
- Version tags (v*)
- Published releases

**Jobs:**
- `build-and-test`: Multi-platform builds and testing
- `build-python`: Python package building
- `build-rust`: Rust crate building
- `build-embedded`: Embedded library artifacts
- `build-docs`: Documentation building
- `publish-python`: PyPI publishing
- `publish-rust`: Crates.io publishing
- `publish-docker`: Container image publishing
- `create-release`: GitHub release creation

**Artifacts:**
- Python wheels and source distributions
- Rust crates
- Embedded library binaries
- Docker images
- Documentation sites

### Security Workflow (security-scan.yml)

**Triggers:**
- Push to main/develop branches
- Pull requests to main branch
- Weekly scheduled scans
- Manual dispatch

**Jobs:**
- `secret-scan`: Secret detection with detect-secrets
- `dependency-scan`: Vulnerability scanning with Safety/Cargo Audit
- `code-analysis`: Static analysis with Semgrep/MyPy
- `codeql-analysis`: GitHub CodeQL analysis
- `embedded-security`: Embedded-specific security checks
- `container-scan`: Container vulnerability scanning
- `supply-chain`: SBOM generation and supply chain analysis
- `security-report`: Comprehensive security reporting

**Artifacts:**
- Security scan reports
- SBOM files
- Vulnerability assessments

### Dependency Update Workflow (dependency-update.yml)

**Triggers:**
- Weekly scheduled updates (Mondays)
- Manual dispatch

**Jobs:**
- `python-dependencies`: Python package updates
- `rust-dependencies`: Rust crate updates
- `github-actions`: GitHub Actions updates
- `security-updates`: Security-focused updates
- `compatibility-check`: Cross-version compatibility testing
- `documentation-update`: Changelog and docs updates
- `notification`: Team notifications

**Artifacts:**
- Updated dependency files
- Compatibility reports
- Automated pull requests

## 🔒 Security Considerations

### Secrets Management
- Use GitHub Secrets for sensitive data
- Rotate tokens regularly
- Use environment-specific secrets
- Never log secret values

### Permissions
- Use minimal required permissions
- Enable OIDC for cloud deployments
- Restrict workflow modifications
- Monitor action usage

### Dependency Security
- Pin action versions with SHA hashes
- Use official actions when possible
- Review third-party actions
- Enable Dependabot for workflow dependencies

## 📊 Monitoring and Metrics

### Workflow Metrics
- Build success rates
- Test execution times
- Security scan results
- Deployment frequencies

### Dashboards
- GitHub Actions usage dashboard
- Security posture tracking
- Dependency update tracking
- Performance trend analysis

## 🔧 Troubleshooting

### Common Issues

**Workflow not triggering:**
- Check trigger conditions
- Verify branch protection rules
- Check Actions permissions
- Review repository settings

**Test failures:**
- Check test environment setup
- Verify dependency installations
- Review resource constraints
- Check for flaky tests

**Security scan failures:**
- Update security baselines
- Review detected vulnerabilities
- Check scan configuration
- Verify tool versions

**Deployment failures:**
- Check deployment credentials
- Verify target environments
- Review deployment scripts
- Check resource availability

### Debug Commands

```bash
# Check workflow syntax
gh workflow view ci.yml

# View workflow runs
gh run list --workflow=ci.yml

# Check workflow logs
gh run view <run-id> --log

# Re-run failed workflows
gh run rerun <run-id>
```

## 📞 Support

For workflow-related issues:

1. Check this documentation first
2. Review GitHub Actions logs
3. Search existing issues with `workflow` label
4. Create new issue with:
   - Workflow name
   - Error message
   - Steps to reproduce
   - Relevant logs

## 🔗 References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Using Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

## ✅ Verification Checklist

After setting up workflows, verify:

- [ ] All workflow files are in `.github/workflows/`
- [ ] Required secrets are configured
- [ ] Branch protection rules are active
- [ ] Actions permissions are set correctly
- [ ] First CI run completes successfully
- [ ] Security scans execute without errors
- [ ] Dependency updates create PRs correctly
- [ ] CD pipeline runs on test release