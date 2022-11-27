from setuptools import setup, find_namespace_packages

setup(
    name="rlearn",
    url="https://github.com/tboulet/Deep-RL-Framework", 
    author="Timothé Boulet",
    author_email="timothe.boulet0@gmail.com",
    
    packages=find_namespace_packages(),

    version="1.0",
    license="MIT",
    description="RLearn is a framework for Deep Reinforcement Learning.",
    long_description=open('README.md').read(),      
    long_description_content_type="text/markdown",  
)