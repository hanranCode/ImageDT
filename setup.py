#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
	name = 'imagedt',
	version = '0.0.19',
	keywords = ('imagedt'),
	description = 'a python lib for neural networks, file and image processing etc. ',
	license = 'Apache License 2.0',

	url = 'https://github.com/hanranCode/ImageDT',
	author = 'pytorch_fans11',
	author_email = 'zhengyzms@163.com',

	packages = find_packages(),
	include_package_data = True,
	platforms = 'any',
	install_requires = [],
)
