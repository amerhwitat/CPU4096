# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately so we can address it before public disclosure.

Preferred contact methods:
- Email: security@your-organization.example (replace with your secure contact)
- Encrypted email: use the project's public PGP key (available in `keys/` or via a secure channel)

When reporting a vulnerability, please include:
- A concise description of the issue
- Steps to reproduce
- Impact assessment (if known)
- Any proof-of-concept code or test cases
- Your contact information for follow-up

We will acknowledge receipt within 48 hours and provide a remediation timeline.

## Disclosure Policy

- We follow coordinated disclosure. We will work with the reporter to fix the issue and will not publicly disclose details until a fix is available or an agreed disclosure date is reached.
- We will credit reporters who agree to be named in release notes.

## Security Practices

- Signing keys and other sensitive secrets are **never** stored in the repository.
- CI secrets are stored in the repository host's secret store and are not exposed in logs.
- Kernel modules and low-level components are subject to static analysis and code review before release.

## Emergency

If you believe a vulnerability is being actively exploited, please mark the email subject as **[EMERGENCY]** and include a phone number for urgent contact.
