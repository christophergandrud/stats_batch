#!/usr/bin/env zsh

echo "Building the package"
pip3 install --upgrade .

echo "Generating and generating coverage report"
coverage run -m pytest

echo "Updating code coverage badge for README"
coverage-badge -o img/coverage.svg -f

echo "Building distribution"
python3 -m build
