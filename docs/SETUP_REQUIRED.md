# Manual Setup Required

Due to GitHub App permission limitations, the following setup steps must be completed manually by repository administrators.

## GitHub Workflows

### Required Files
Copy the following workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow files
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### Required Secrets
Configure the following secrets in GitHub repository settings:

- `PYPI_API_TOKEN`: Token for publishing to PyPI
- `CRATES_IO_TOKEN`: Token for publishing to crates.io  
- `SLACK_WEBHOOK_URL`: Webhook URL for Slack notifications (optional)
- `CODECOV_TOKEN`: Token for code coverage reporting (optional)

### Required Permissions
Enable the following GitHub Actions permissions:
- Actions: Read and write
- Contents: Read and write  
- Metadata: Read
- Pull requests: Read and write
- Issues: Read and write

## Branch Protection Rules

Configure branch protection for `main` branch:

```yaml
Required status checks:
  - test-python (3.8, 3.9, 3.10, 3.11)
  - test-rust
  - test-cpp
  - security-scan
  - integration-test

Require branches to be up to date: true
Require linear history: true
Restrict pushes that create files: false
```

## Repository Settings

### Topics
Add the following topics to improve discoverability:
- `machine-learning`
- `audio-processing`
- `embedded-systems`
- `edge-ai`
- `neural-networks`
- `rust`
- `python`
- `liquid-networks`

### Repository Description
```
Edge-efficient Liquid Neural Network models for always-on audio sensing with 10× power reduction
```

### Homepage URL
```
https://liquid-audio-nets.readthedocs.io
```

## Issue Templates

Create `.github/ISSUE_TEMPLATE/` directory with:
- `bug_report.yml`
- `feature_request.yml`
- `performance_issue.yml`
- `embedded_support.yml`

## Pull Request Template

Create `.github/PULL_REQUEST_TEMPLATE.md` with checklist for:
- Code quality requirements
- Testing requirements
- Documentation updates
- Breaking change notifications

## Security Configuration

### Dependabot
Enable Dependabot security updates for:
- Python (pip)
- Rust (cargo)
- GitHub Actions

### Code Scanning
Enable CodeQL analysis for:
- Python
- C/C++
- JavaScript (for documentation)

## Monitoring & Analytics

### Repository Insights
Enable repository insights and configure:
- Code frequency analysis
- Contributor statistics
- Traffic analytics

### Performance Monitoring
Set up monitoring for:
- Build times
- Test execution times
- Binary size tracking
- Performance regression detection

## Community Health

### Files to Review
Ensure the following community files are properly configured:
- [x] `CODE_OF_CONDUCT.md`
- [x] `CONTRIBUTING.md`
- [x] `LICENSE`
- [x] `SECURITY.md`
- [x] `README.md`

### Labels
Create repository labels for:
- `bug`, `enhancement`, `documentation`
- `embedded`, `performance`, `security`
- `good first issue`, `help wanted`
- `python`, `rust`, `cpp`
- `breaking-change`, `needs-testing`

## Release Configuration

### Semantic Release
Configure semantic-release for automated versioning:
```json
{
  "branches": ["main"],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    "@semantic-release/github"
  ]
}
```

### Version Tagging
Follow semantic versioning (SemVer):
- Major: Breaking changes
- Minor: New features
- Patch: Bug fixes

## Documentation Hosting

### GitHub Pages (if using)
Configure GitHub Pages to serve documentation from:
- Source: GitHub Actions
- Branch: `gh-pages`
- Custom domain: `liquid-audio-nets.dev` (optional)

### External Hosting (recommended)
Set up documentation hosting on:
- Read the Docs
- Netlify
- Vercel

## Third-Party Integrations

### Code Quality
- **Codecov**: Code coverage reporting
- **Sonar**: Code quality analysis
- **LGTM**: Security analysis

### Dependency Management
- **Renovate**: Automated dependency updates
- **Snyk**: Security vulnerability scanning
- **David**: Dependency status monitoring

### Communication
- **Slack/Discord**: Community chat integration
- **GitHub Discussions**: Community Q&A
- **Mailing Lists**: Development announcements

## Completion Checklist

### Core SDLC Implementation ✅
- [x] Project foundation and documentation structure
- [x] Development environment and tooling configuration  
- [x] Comprehensive testing infrastructure
- [x] Build system and containerization
- [x] Monitoring and observability setup
- [x] Workflow documentation and templates
- [x] Metrics tracking and automation scripts
- [x] Integration and final configuration

### GitHub Setup (Manual Required)
- [ ] GitHub workflows copied and configured
- [ ] Repository secrets configured
- [ ] Branch protection rules enabled
- [x] Issue and PR templates created
- [ ] Security features enabled
- [x] Community health files reviewed
- [ ] Third-party integrations configured
- [ ] Documentation hosting configured
- [ ] Release process tested

### Verification Steps
- [ ] Run `make test` to verify testing infrastructure
- [ ] Execute `python scripts/automation/repository_maintenance.py` for health check
- [ ] Validate Docker builds with `docker build -t liquid-audio-nets .`
- [ ] Test monitoring stack with `cd monitoring && docker-compose up`
- [ ] Verify automation scripts work correctly

## Support

For questions about this setup process:
1. Check the [CONTRIBUTING.md](../CONTRIBUTING.md) guide
2. Open an issue with the `setup` label
3. Contact the maintainers directly

---

*This document should be updated as additional manual setup requirements are identified.*