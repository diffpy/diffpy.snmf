---
name: Release checklist
about: Checklist before releasing on PyPI and conda-forge
title: ""
labels: ""
assignees: ""
---

**Release checklist for GitHub contributors**

- [ ] Have you checked that all PRs/issues are resolved for the release version?
- [ ] Are you using the latest version of [cookiecutter](https://github.com/Billingegroup/cookiecutter)? Ensure `.github/workflows/build-wheel-release-upload.yml` is available.
- [ ] Have you checked that all the badges on the README are passing?
- [ ] Have you locally rendered the documentation and confirmed that all pages, including tutorials, are displaying correctly?
- [ ] Have you checked and updated the installation instructions in the documentation and on the website (e.g., diffpy.org)?
- [ ] Have you checked for any grammar or typos in the documentation and code?
- [ ] Have you verified the license information? If you are unsure, please comment below.

Please mention @sbillinge when you are ready for release. Include any additional comments necessary, such as version information and details about the pre-release.