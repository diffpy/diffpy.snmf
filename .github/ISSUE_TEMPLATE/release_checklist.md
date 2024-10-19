---
name: Release
about: Checklist and communication channel for PyPI and GitHub release
title: "Ready for <version-number> PyPI/GitHub release"
labels: "release"
assignees: ""
---

**Release checklist for GitHub contributors**

- [ ] All PRs/issues are resolved for the release version.
- [ ] The latest version of [cookiecutter](https://github.com/Billingegroup/cookiecutter) is used. Ensure `.github/workflows/build-wheel-release-upload.yml` is available.
- [ ] All the badges on the README are passing.
- [ ] Locally rendered documentation contains all appropriate pages, including API references, tutorials, and extensions.
- [ ] Installation instructions in the documentation and on the website (e.g., diffpy.org) are checked and updated.
- [ ] Grammar and writing quality have been checked.
- [ ] License information is verified. If you are unsure, please comment below.

Please mention @sbillinge when you are ready for release. Include any additional comments necessary, such as version information and details about the pre-release.
