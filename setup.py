from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="rlearn",
    version="1.0",
    author="Timothé Boulet",
    author_email="timothe.boulet0@gmail.com",
    url="https://github.com/tboulet/Deep-RL-Framework", 
    
    packages=find_namespace_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'rlearn-train=rlearn.train:main'
            'rlearn-enjoy=rlearn.enjoy:main' # TODO : add enjoy
        ]
    },

    license="MIT",
    description="RLearn is a framework for Deep Reinforcement Learning.",
    long_description=open('README.md').read(),      
    long_description_content_type="text/markdown",  
)