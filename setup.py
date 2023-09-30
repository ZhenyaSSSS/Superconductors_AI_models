from setuptools import setup, find_packages

setup(
  name = 'model',
  version = '0.1',
  description = 'My ML model', 
  py_modules = ['model'],
  install_requires = [
    'torch',
    'pytorch_lightning',
    'timm',
  ] 
)
