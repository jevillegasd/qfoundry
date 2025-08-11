from distutils.core import setup

# Deprecated: use pyproject.toml for builds and installs.
# This file remains only for compatibility with old workflows.
# Install from the repository root:
#   pip install -e .
raise SystemExit(
    "Use 'pip install -e .' from the repository root (pyproject.toml). This setup.py is deprecated."
)

setup(name="qfoundry", version="0.0.1", packages=["qfoundry"])
