from setuptools import setup, find_packages

setup(
    name='botied',
    version='v0.10',
    author='Ji Won Park, Natasa Tagasovska',
    author_email='park.ji_won@gene.com',
    packages=find_packages(),
    description='Multi-objective Bayesian optimization with tied multivariate ranks',
    long_description=open("README.md").read(),
    long_description_content_type='text/x-rst',
    url='https://github.com/jiwoncpark/botied',
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose'],
)
