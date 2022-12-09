#!/bin/bash

# retrive AITemplate 0.1 release from git and install
cd /
git clone https://github.com/supunab/AITemplate.git
cd AITemplate/python
python3 setup.py bdist_wheel
pip3 install --no-input /AITemplate/python/dist/*.whl
