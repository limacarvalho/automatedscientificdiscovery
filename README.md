![MPI IPP Logo](img/ipp_logo.png)

# Automated Scientific Discovery (ASD)

Automated Scientific Discovery is an open-source Machine Learning/Bayesian Statistics tool aimed at discovering predictability and useful information in scientific databases/datasets. This tool is designed for scientists without AI domain knowledge and 
focuses on a limited signal domain (scientific signals/parameters) and a limited purpose (finding predictability).

## Features

-   Utilizes MLP neural networks, Kernel density estimation, and Gaussian processes
-   Employs Bayesian methods for optimizing hyperparameters and calculating uncertainties
-   Novelty detection for analyzing points and detecting novel features
-   Sensitivity analysis for identifying relevant predictive parameters
-   Built upon widely used tools such as Ray.io, Torch, XGBoost, Streamlit, Sklearn, Scipy, Optuna, etc
-   Tested on datasets from nuclear fusion and climate science

## Milestones

1.  **Automatic Model Training**: Demonstrate fully automatic training of 2-3 machine learning models for predicting certain phenomena from given indicators. Models should handle uncertainties, including errors on variables and 
predictive errors.
2.  **Automated Clustering**: Demonstrate fully automated clustering in scientific datasets.
3.  **Visualization Techniques**: Demonstrate visualization techniques for dataset exploration, and specialized visualizations for identifying potential problems in datasets used for machine learning model training.
4.  **Cross-domain Application**: Demonstrate the system working on at least two datasets from different research fields: nuclear fusion and climate science.

## Future Extensions

-   Application to other research fields in the Helmholtz group
-   Multi-Cloud backend support for multiprocessing/parallel jobs
-   Higher level discovery: automated searches for predictability or clusters in whole databases
-   Automated design of experiments (using Bayesian optimization)
-   Discovering symbolic expressions (through Bayesian Program Synthesis)
-   Phase space analysis
-   Algorithmic complexity theory

## Supported Platforms / Requirements
- Linux x86_64 (bash or zsh)
- Docker Engine 23.0.0+
- 50GB+ local disk space available
- Core i5+ (or comparable processor) 16GB+ RAM
- No Apple silicon (M1, M2, etc) supported (further testing and adjustments needed))
- No GPU support currently (still in development)

## Getting Started

1. Open up your terminal of choice and run the following command:
```bash
curl https://raw.githubusercontent.com/limacarvalho/automatedscientificdiscovery/main/src/infrastructure/containers/asd_local_container/asd.sh -o asd.sh && bash asd.sh
```
2. Wait for the container creation/build process to finish (it can take from a few minutes up to an hour depending on local resources available like CPU and mem), you will be able to get the access instructions when the execution 
finishes up successfully:

```text
+----------------------------------------------------------------------------+
|             +++ The ASD Container is currently running +++                 |
+-----------------------------------------+----------------------------------+
| To stop the container run:              | docker container stop asd        |
| To delete the container run:            | docker container rm asd          |
+-----------------------------------------+----------------------------------+

...

  You can now view your Streamlit app in your browser.

  Network URL: http://172.17.0.2:80
  External URL: http://18.156.6.181:80
```

## Main contributors
- Jakob Svensson (Specialising in Bayesian modelling and AI applied to scientific problems. Seed eScience Research Ltd. PhD physics. Formerly Max Planck Inst. for Plasma physics) ([Researchgate 
link](https://www.researchgate.net/profile/Jakob-Svensson-5))
- Michael Koehn (Principal AI Consultant) ([Koehn AI](https://www.koehn.ai/en/))
- David Winnekens (AI Consultant) ([Koehn AI](https://www.koehn.ai/en/))
- Kay Eckelt (Senior Data-Science & Research Consultant, Nuclear Fusion) ([Linkedin](https://www.linkedin.com/in/kay-eckelt-phd/?originalSubdomain=es))
- Wasif Masood (Senior Data Science) ([EmpirischTech](https://empirischtech.at/))
- Gerrit Tombrink (Data Scientist | Data Visualization) ([GEOLINKED](https://geolinked.de/))
- João Carvalho (Software / Cloud Engineer) (limacarvalho.com)


## Contributing

We welcome contributions to improve and extend the project. Please follow the guidelines in CONTRIBUTING.md to submit your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
