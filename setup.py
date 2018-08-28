#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
	name = 'imagedt',
	version = '0.0.3',
	keywords = ('imagedt'),
	description = 'a python lib for neural networks, file and image processing etc. ',
	long_description="##[Details](https://github.com/hanranCode/ImageDT)",
	long_description_content_type="text/markdown",
	license = 'Apache License 2.0',

	url = 'https://github.com/hanranCode/ImageDT',
	author = 'pytorch_fans11',
	author_email = 'zhengyzms@163.com',

	packages = find_packages(),
	include_package_data = True,
	platforms = 'any',
	install_requires = [],
)
