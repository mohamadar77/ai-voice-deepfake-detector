# Security Policy

## Scope

This repository contains research code and a local demonstration service for voice
spoof detection. It is not an identity-verification product or a production security
control. A model score must not be used as proof of authenticity or as the sole basis
for consequential decisions.

Security reports may concern audio upload handling, request limits, temporary-file
cleanup, unsafe deserialization/model loading, dependency vulnerabilities, container
configuration, secret exposure or privacy leaks.

## Supported versions

There is currently no published release or supported-version matrix. Until a release
is tagged, report issues against the current `main` branch and include the affected
commit when known.

## Reporting a vulnerability

**Do not create a public GitHub issue or pull request for an exploitable vulnerability.**
Use GitHub's private vulnerability reporting for this repository (Security → Advisories →
Report a vulnerability) when it is enabled. If that option is unavailable, contact the
repository owner privately through their GitHub profile and request a secure reporting
channel.

Include a concise description, affected commit/path, impact, prerequisites and safe
reproduction steps. Redact tokens, personal data and local paths. Do not attach voice
recordings, datasets, trained weights or other sensitive artifacts unless the maintainer
has explicitly arranged a safe transfer.

Please allow the maintainer to investigate and coordinate a fix before public
publication. No response-time or service-level commitment is made by this policy.

## Privacy

Treat all uploaded voice recordings as sensitive personal data. Do not store raw audio
in logs, issues, pull-request comments or CI artifacts. The application is expected to
process uploads temporarily and avoid retaining them after the request completes or
fails.
