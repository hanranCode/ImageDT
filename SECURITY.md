# Security policy

The maintained branch is `main`. Security fixes target the current Python 3 release line; old Python 2 environments and removed third-party backends are not maintained security targets. Candidate releases should be evaluated against the documented compatibility limits before production use.

Please do not publish credentials, private datasets or working exploit details in a public issue. Use GitHub's **Report a vulnerability** option in this repository's Security tab when available. If private reporting is unavailable, contact the maintainer through the author contact in the package metadata and ask for a private reporting channel before sending sensitive details. Do not send model files or personal data until the recipient has agreed to receive them.

A useful report includes affected version, module/function, required conditions, expected impact, and a minimal reproduction using synthetic data. No response-time or bounty program is promised.

Model conversion and weight loading process caller-supplied files. Only load trusted model/weight artifacts; the legacy NumPy weight format uses pickle-compatible loading. ImageDT is an offline utility library, not a sandbox for hostile models. File deletion and rename tools should run with permissions limited to the intended dataset.
