# Security Guidelines

These guidelines apply to every employee and contractor at Halden Systems and to every device used for company work.

## Accounts and passwords

Use the company password manager, Vaultline, for all work credentials. Passwords must be at least 14 characters long and unique per service. Multi-factor authentication is mandatory for email, source control, the cloud console and the HR system. Hardware security keys are required for anyone with production access; SMS codes are not accepted as a second factor.

## Devices

Company laptops must have full-disk encryption enabled and must lock automatically after 5 minutes of inactivity. Operating system security updates must be installed within 7 days of release. Personal devices may be used for email and chat only, through the managed mobile app, and never for source code or customer data.

## Customer data

Customer data must never be copied to local machines, personal storage or third-party tools that are not on the approved vendor list. Production databases are accessed only through the audited bastion host, and every access is logged.

## Reporting incidents

Report a suspected security incident, such as a lost laptop, a phishing email you clicked, or leaked credentials, to security@halden.example within 1 hour of noticing it, or in the #security-incidents chat channel. Lost or stolen devices are wiped remotely, so report them immediately even if you hope to find them.

## Access reviews

Managers review their team members' access to production systems every quarter. Access that is no longer needed is removed within 2 working days of the review.
