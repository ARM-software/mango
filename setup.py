import setuptools

setuptools.setup(
    name='arm-mango',
    version='1.0.0',
    author='Arm Research',
    author_email='mohit.aggarwalh@arm.com',
    description='parallel Bayesian optimization over complex search spaces',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ARM-software/mango',
    packages=['mango', 'mango.optimizer', 'mango.domain'],
    include_package_data=True,
    package_data={
        '': ['*.cfg'],
    },
    install_requires=[
        'numpy>=1.17.0',
        'scipy>=1.4.1',
        'scikit_learn>=0.21.3',
        'tqdm>=4.36.1',
        'attrdict>=2.0.1',
        'dataclasses'
    ],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
