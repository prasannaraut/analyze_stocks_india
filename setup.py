from setuptools import setup

setup(
    name='analyze_stocks_india_dir',
    version='1.0.0',
    description='A python package to download historic stock prices from NSE website and process them to find best stock to purchase',
    author='Prasanna Raut',
    author_email='prasannaraut36@gmail.com',
    packages=['analyze_stocks_india_dir'],
    install_requires=[
        'numpy>=2.0.0',
        'pandas>=2.2.2',
        'nltk>=3.8.1',
        'plotly>=5.22.0',
        'requests>=2.32.3',
        'bs4',
        'tenacity>=8.3.0',
    ],
)
