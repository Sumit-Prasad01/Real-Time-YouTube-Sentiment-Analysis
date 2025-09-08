from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    
    requirements_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()

                if requirement and requirement != '-e .':
                    requirements_lst.append(requirement)                   
            
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return requirements_lst

setup(
    name="Youtube Sentiment Insights",
    version="0.0.1",
    author="Sumit Prasad",
    author_email="",
    packages=find_packages(),
    install_requires=get_requirements()
)