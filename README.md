# Automated Scientific Discovery - Python

![OS](https://img.shields.io/badge/OS-Linux-red?style=flat&logo=linux)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python%203.8-1f425f.svg?logo=python)](https://gitlab.com/automatedscientificdiscovery)
[![Docker](https://img.shields.io/badge/Docker-available-green.svg?style=flat&logo=docker)](https://gitlab.com/automatedscientificdiscovery)
[![Maintenance](https://img.shields.io/badge/Maintained-yes-green.svg)](https://gitlab.com/automatedscientificdiscovery)
[![GitHub](https://img.shields.io/github/license/emalderson/ThePhish)](https://gitlab.com/automatedscientificdiscovery)
[![Documentation](https://img.shields.io/badge/Documentation-complete-green.svg?style=flat)](https://gitlab.com/automatedscientificdiscovery)

⚡ Python-related code ⚡

## Access/Clone repository
- Make sure you have git installed on your local machine
  - If not get the latest version from https://git-scm.com/downloads or install using the appropriate package manager (eg. brew, choco, apt, etc)
- Run ```git clone https://gitlab.com/automatedscientificdiscovery/python-asd.git``` from a directory of your choice in your local machine  (if prompted enter your GitLab credentials)
- Access the newly download folder
- Run ```git checkout develop```
- Create your own project-related folder (eg. predictability, complexity, relevance, etc)
-  Upload your code to the newly created folder
- Make sure to change directories to the git repository main directory (eg. python-asd)
- Run ```git add *```
- Run ```git status```
- Verify that the files you want to upload/push are listed in the command output, and that temp files or cache files are not (eg. pyc, wheel, .DS_Store, etc)
- Run ```git commit -m "REPLACE HERE WITH A MEANINGFUL MESSAGE"```
- Run ```git push``` (if prompted, enter your GitLab credentials)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

As much as possible, try to follow the following rules/guidelines during development:
- Try to use Python3.7.7 or latest (To use Ray in the Cluster Python=3.7.7 is needed)
- Document code whenever feasible using docstrings and comments
- Do not push code containing API Keys, secrets or any other sensitive information
- Implement timeouts, retries, and backoff with jitter whenever dependent on external APIs
- [Type-hints](https://docs.python.org/3/library/typing.html) are recommended
- Try to follow code formatting rules as described in the PEP8 whenever possible. Use the python module [black](https://github.com/psf/black)
- Push code to "develop" branch and when validate initiate a pull request to "master"
- Always make sure to include the requirements.txt file (can be generated from your command line using pip freeze > requirements.txt) or the conda environment list of packages (conda env export > environment. yml) 
- Rules for Python variables:
  - A variable name must start with a letter or the underscore character
  - A variable name cannot start with a number
  - A variable name can only contain alpha-numeric characters and underscores (A-z, 0-9, and _ )
  - Variable names are case-sensitive (age, Age and AGE are three different variables)
- Rules for Python classes:
  - Start each word with a capital letter. Do not separate words with underscores. This style is called camel case or pascal case.	
- Do not use hardcoded filesystem references in your code like '/Users/joaocarvalho/Downloads/code_example.py' instead use the pathlib module or any other similar module to deal with filesystem paths
- Use try/except/else/finally blocks in your code to allow proper error handling and retries
- Use the logging module instead of 'print' statements (https://blog.sentry.io/2022/07/19/logging-in-python-a-developers-guide/)
- Use reStructured Text for Docstrings (Sphinx Style) https://www.datacamp.com/tutorial/docstrings-python#sphinx-style:
  ```class Vehicle(object):
    '''
    The Vehicle object contains lots of vehicles
    :param arg: The arg is used for ...
    :type arg: str
    :param `*args`: The variable arguments are used for ...
    :param `**kwargs`: The keyword arguments are used for ...
    :ivar arg: This is where we store arg
    :vartype arg: str
    '''


    def __init__(self, arg, *args, **kwargs):
        self.arg = arg

    def cars(self, distance, destination):
        '''We can't travel a certain distance in vehicles without fuels, so here's the fuels

        :param distance: The amount of distance traveled
        :type amount: int
        :param bool destinationReached: Should the fuels be refilled to cover required distance?
        :raises: :class:`RuntimeError`: Out of fuel

        :returns: A Car mileage
        :rtype: Cars
        '''  
        pass
    ```
- Please make sure to update tests as appropriate.

## License
[MIT License](https://en.wikipedia.org/wiki/MIT_License)

