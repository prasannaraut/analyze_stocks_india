from setuptools import setup

setup(
    name='analyze_stocks_india',
    version='1.0.1',
    description='A python package to download historic stock prices from NSE website and process them to find best stock to purchase',
    author='Prasanna Raut',
    author_email='prasannaraut36@gmail.com',
    packages=['analyze_stocks_india', 'analyze_stocks_india_sample_scripts'],
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.1.5",
        "nltk>=3.6.7",
        "plotly>=5.18.0",
        "requests>=2.27.1",
        "bs4",
        "tenacity>=8.2.2",
    ],
)
