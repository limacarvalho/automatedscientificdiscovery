#!/bin/bash

# +------------------------------+----------------------------------+
# |              ASD Container runtime setup script                 |
# +------------------------------+----------------------------------+
# | Version                      | 1.1                              |
# | Language                     | Linux Bash                       |
# | Platform                     | x86_64                           |
# | Input Parameters             | None                             |
# | GPU / non-GPU support        | No GPU Support                   |
# | Runs on Docker Image         | tensorflow/tensorflow:2.13.0     |
# +------------------------------+----------------------------------+

# General OS dependencies installation
apt update -y
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
add-apt-repository ppa:git-core/ppa -y
apt update -y
apt install git -y
apt install nano pciutils r-base r-base-dev libgmp-dev libmpfr-dev -y

# Creates asd directory if not exist
mkdir -p /opt/asd

# Python dependencies installation
python -m pip install Cython
python -m pip install GenDoc -U
python -m pip install git+https://github.com/h2oai/datatable.git
python -m pip install bokeh==2.4.3

# Debug information

echo -e "\n+++++++++++++++++++++++++++++++++++++++++++++++\n" | tee -a $asd_init_debug_file
python --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n+++++++++++++++++++++++++++++++++++++++++++++++\n" | tee -a $asd_init_debug_file
gcc --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n+++++++++++++++++++++++++++++++++++++++++++++++\n" | tee -a $asd_init_debug_file
nvcc --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n+++++++++++++++++++++++++++++++++++++++++++++++\n" |  tee -a $asd_init_debug_file
lscpu 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n+++++++++++++++++++++++++++++++++++++++++++++++\n" |  tee -a $asd_init_debug_file
lsmem 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n+++++++++++++++++++++++++++++++++++++++++++++++\n" |  tee -a $asd_init_debug_file

cat << EOF > /root/requirements.txt
absl-py
aiohttp
aiohttp-cors
aiosignal
aiosqlite
alembic
altair
anyio
apprise
asgi-lifespan
astunparse
async-timeout
asyncer
asyncpg
attrs
autopage
bayesian-optimization
bleach
blessed
blinker
boto3
botocore
cachetools
captum
certifi
cffi
charset-normalizer==2.0.12
choldate @ git+https://github.com/jcrudy/choldate.git@0d92a523f8da083031faa0eb187a7b0f287afe69
click
cliff
cloudpickle
cmaes
cmd2
colorama
colorcet
colorful
colorlog
commonmark
contourpy
coolname
croniter
cryptography
cvxpy
cycler
dcor
deap
decorator
distlib
dm-tree
docker
ecos
entrypoints
eplot
fastapi
ffmpeg
filelock
flatbuffers
fonttools
frozenlist
fsspec
gast
gitdb
GitPython
glmnet
google-api-core
google-auth
google-auth-oauthlib
google-pasta
googleapis-common-protos
graphviz
greenlet
griffe
grpcio
h11
h2
h5py
holoviews
hpack
htmlmin
httpcore
httpx
hyperframe
hypertools
idna
ImageHash
importlib-metadata==4.13.0
Jinja2
jmespath
joblib
jsonpatch
jsonpointer
jsonschema
keras
kiwisolver
knockpy
kubernetes
libclang
lightgbm
lightning-utilities
llvmlite
Mako
Markdown
MarkupSafe
matplotlib
msgpack
multidict
multimethod
networkx
numba
oauthlib
opencensus
opencensus-context
opt-einsum
optuna
orjson
osqp
packaging
pandas
pandas-profiling
panel
param
pathspec
patsy
pbr
pendulum
phik
pickleshare
Pillow
platformdirs
plotly
ppca
prefect
prefect-ray
prettytable
prometheus-client
protobuf
psutil
py-spy
pyarrow
pyasn1
pyasn1-modules
pycparser
pyct
pydantic
pydeck
pyecharts
Pygments
Pympler
pynndescent
pyparsing
PypeR
pyperclip
pyrsistent
python-dateutil
python-slugify
pytorch-lightning
pytz
pytz-deprecation-shim
pytzdata
pyviz-comms
PyWavelets
PyYAML
qdldl
ray[default]==2.6.1
ray[tune]
readchar
requests==2.27.0
requests-oauthlib
rfc3986==1.5.0
rich
rsa
s3transfer
scikit-dimension
scikit-learn
scipy==1.8.1
scs
seaborn
semver
shap
simplejson
six
skorch
slicer
smart-open
smmap
sniffio
SQLAlchemy
starlette==0.22.0
statsmodels
stevedore
stopit
streamlit
streamlit-pandas-profiling
swig
tabulate
tangled-up-in-unicode
tenacity
termcolor
text-unidecode
threadpoolctl
toml
toolz
torch
torchmetrics
torchvision
tornado
TPOT
tqdm
tune-sklearn
typeguard
typer
typing_extensions
tzdata
tzlocal
umap-learn
update-checker
urllib3==1.26.13
uvicorn
validators
virtualenv
visions
watchdog
wcwidth
webencodings
websocket-client
Werkzeug
wrapt
xgboost
xyzservices
yarl
zipp
EOF

# Python requirements installation needed for ASD
python -m pip install -r /root/requirements.txt

python -m pip freeze > /opt/asd/asd-requirements-with-versions.txt

# Setting Python PATH
echo 'PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"' >> /etc/environment

# Protobuf settings
curl -o /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/builder.py https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py

# Installation of R dependencies
Rscript -e 'install.packages(c("base", "base64enc", "bit", "bit64", "boot", "class", "cluster", "codetools", "compiler", "CVXR", "datasets", "dbscan", "digest", "ECOSolveR", "evaluate", "fastcluster", "fastmap", "foreign", "glue", "gmp", "graphics", "grDevices", "grid", "highr", "htmltools", "htmlwidgets", "jsonlite", "KernSmooth", "knitr", "labdsv", "lattice", "magrittr", "maotai", "MASS", "Matrix", "mclustcomp", "methods", "mgcv", "minpack.lm", "nlme", "nnet", "osqp", "parallel", "R6", "RANN", "rbibutils", "Rcpp", "RcppArmadillo", "RcppDE", "RcppDist", "RcppEigen", "Rcsdp", "Rdimtools", "rdist", "Rdpack", "rgl", "rlang", "Rmpfr", "rpart", "RSpectra", "Rtsne", "scatterplot3d", "scs", "shapes", "spatial", "splines", "stats", "stats4", "stringi", "stringr", "survival", "tcltk", "tools", "utils", "xfun", "yaml"), repos="https://cloud.r-project.org")' | tee -a /opt/asd/$CONTAINER_NAME-r-dependencies-installation-log.txt

# Clone the ASD git repository (excluding the 'container folder')
git clone https://github.com/limacarvalho/automatedscientificdiscovery.git /opt/asd/python-asd
cd /opt/asd/python-asd

# Forces the correct apt sources.list without duplicate entries
cat << EOF > /etc/apt/sources.list
# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.

deb http://archive.ubuntu.com/ubuntu/ focal main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal main restricted universe multiverse

deb http://archive.ubuntu.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal-updates main restricted universe multiverse

deb http://archive.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse

deb http://archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse

deb http://archive.canonical.com/ubuntu focal partner
deb-src http://archive.canonical.com/ubuntu focal partner
EOF

exit 0