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

As much as possible try to follow the following rules/guidelines during development:
- Try to use Python3.8 or latest
- Document code whenever feasible
- Do not push code containing API Keys, secrets or any other sensitive information
- Implement timeouts, retries, and backoff with jitter whenever dependent on external APIs
- [Type-hints](https://docs.python.org/3/library/typing.html) are recommended
- Try to follow code formating rules as described in the PEP8 whenever possible. Maybe using [black](https://github.com/psf/black)?
- Push code to "develop" branch and when validate initiate a pull request to "master"

Please make sure to update tests as appropriate.

## License
[MIT License](https://en.wikipedia.org/wiki/MIT_License)

