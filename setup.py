from setuptools import setup, find_packages

setup(
  name = 'ML',
  version = '0.1',
  description = 'My ML model', 
  py_modules = ['ML'],
  install_requires = [
    'torch',
    'pytorch_lightning',
    'timm',
  ] 
)
