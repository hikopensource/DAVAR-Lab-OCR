from setuptools import find_packages, setup


def readme():
    with open('../readme.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = './davarocr/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':
    setup(
        name='davarocr',
        version=get_version(),
        description='DAVAR Lab @ Hikvision Research Institute',
        long_description=readme(),
        keywords='computer vision',
        packages=find_packages(),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        url='https://davar-lab.github.io/',
    )
