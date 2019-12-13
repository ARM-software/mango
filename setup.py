import setuptools

setuptools.setup(
    name='mango',
    version='1.0.0',
    author='IoT Services Research',
    author_email='iotresearch@arm.com',
    description='parallel optimization over complex search spaces',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://gitlab.com/arm-research/isr/mango',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.cfg'],
    },
    install_requires=[
        'numpy==1.17.0',
        'scipy==1.3.1',
        'scikit_learn==0.21.3',
        'tqdm==4.36.1'
    ],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Posix :: Linux',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Data Analytics',
        'Intended Audience :: Developers',
        'Intended Audience :: Data Science',
        'Intended Audience :: Machine Learning',
        'Intended Audience :: Information Technology',
    ],
)
