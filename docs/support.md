---
title: Future support
---

Scientific software is often developed as part of a project,
and once the project ends, the maintenance status is left undefined.
This document tries to clarify the future support for `ennemi`.

# Roadmap
You can see the plans for future versions on the
[Milestones page on GitHub](https://github.com/polsys/ennemi/milestones).

Please submit an issue if there is something you would like to see fixed or added.


# Future development

`ennemi` has been developed as part of a fixed-period project
(civilian service of the primary author at INAR).
With version 1.0.0, all originally envisioned major features are complete.

Future development consists of three categories:

- Minor maintenance tasks (bug fixes, dependency/Python upgrades)
- Minor features based on user feedback
- Contributions by others, or as part of new projects

Put simply, even though the most active development will slow down,
the package **will be maintained** for the foreseeable future.
Larger new features may require your contributions, but smaller quality-of-life
fixes will be done (given that somebody reports the issue!).


# Version numbering

`ennemi` follows [semantic versioning](https://semver.org).
This means that the version number is always of the format `X.Y.Z`.

- `X` is the major version.
  It will be increased only when the behavior or usage changes in a breaking way;
  that is, when you might have to modify your code to work with the new version.
  We hope to never bump this to `2`.
- `Y` is the minor version.
  It will be increased when new features are added.
  We may also drop support for older Python, NumPy, or SciPy versions here.
  We may remove features, but only after marking them as deprecated
  for at least one minor version.
- `Z` is the patch version.
  It will be increased when bugs are fixed, but no new features are added.
  We will not remove any functionality in patch versions;
  it is always safe and possible to upgrade to the latest patch version.

Some examples (not necessary real!):

- `1.0.0 -> 1.0.1` would only contain bug fixes.
- `1.0.1 -> 1.1.0` would contain new features and bug fixes.
  It might also drop support for older Python or NumPy versions.
  Users of older Python versions would remain on `1.0.1`.
- `1.1.0 -> 2.0.0` would break some of your code.
  It might remove features, and therefore you should be careful when upgrading.

We support only the latest patch versions of the releases.
Previous minor releases may receive bug fixes if necessary,
but the threshold will be decided on a case-by-case basis.


# Python and NumPy support

Support for older versions of Python, NumPy and SciPy may be removed on minor or major versions.
Lower versions of `ennemi` will remain available for users of those.

In the long term, we aim to follow the
[NumPy deprecation policy](https://numpy.org/neps/nep-0029-deprecation_policy.html).
However, there are some deviations:

- `ennemi 1.0.0` requires at least NumPy 1.17.
  This satisfies the "last three minor versions" rule, but not the
  "all versions released in prior 24 months" rule.
  Support for NumPy 1.17 will be kept long enough to satisfy both criteria.
- `ennemi 1.0.0` supports Python 3.6, even though it would not be necessary.
  There are still sufficiently many users on 3.6 to warrant the support.
  The support will be dropped in a future minor release.
