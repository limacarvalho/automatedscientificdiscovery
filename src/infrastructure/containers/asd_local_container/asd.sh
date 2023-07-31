#!/bin/bash

IMAGE_NAME="asd"
IMAGE_VERSION="1.1"
CONTAINER_NAME="asd"
HTTP_PORT="80"

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found. Please install Docker."
    exit
fi

# ASD Folder
asd_folder=~/.asd_container
mkdir -p $asd_folder

# Create Dockerfile locally
cat << EOFMAIN > $asd_folder/Dockerfile
FROM tensorflow/tensorflow:2.13.0
LABEL "com.asd.project"="Automated Scientific Discovery"
LABEL version=$IMAGE_VERSION
LABEL description="The Automated Scientific Discovery project is a Python module/app that automatically discovers hidden relationships in the measurement data."
COPY . /opt/asd/
RUN chmod +x /opt/asd/setup_init.sh && /opt/asd/setup_init.sh
ENV PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"
EXPOSE $HTTP_PORT/tcp
RUN chmod +x /opt/asd/asd_exec.sh
CMD /opt/asd/asd_exec.sh
EOFMAIN

# Create setup_init file locally
asd_init_debug_file="/opt/asd/$CONTAINER_NAME-build-versions.txt"
cat << EOFMAIN > $asd_folder/setup_init.sh
#!/bin/bash

# +------------------------------+----------------------------------+
# |              ASD Container runtime setup script                 |
# +------------------------------+----------------------------------+
# | Version                      | $IMAGE_VERSION                              |
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
apt install nano wget tar zip unzip screen less jq build-essential git gcc make openssl libssl-dev pciutils libffi-dev zlib1g-dev r-base r-base-dev libgmp-dev libmpfr-dev -y

# Install nodejs
curl -sL https://deb.nodesource.com/setup_18.x | bash -
apt install -y nodejs

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Install tailscale
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list | tee /etc/apt/sources.list.d/tailscale.list  
apt update
apt install tailscale -y

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

exit 0
EOFMAIN

# Create asd_exec file locally
asd_init_debug_file="/opt/asd/$CONTAINER_NAME-start-versions.txt"
cat << EOFMAIN > $asd_folder/asd_exec.sh
#!bin/bash

# +------------------------------+----------------------------------+
# |                 ASD Container runtime script                    |
# +------------------------------+----------------------------------+
# | Version                      | $IMAGE_VERSION                              |
# | Language                     | Linux Bash                       |
# | Platform                     | x86_64                           |
# | Input Parameters             | None                             |
# | GPU / non-GPU support        | No GPU Support                   |
# | Runs on Docker Image         | tensorflow/tensorflow:2.13.0     |
# +------------------------------+----------------------------------+

# Debug information
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
echo -e Execution time: $(date -u) | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
python --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
gcc --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" |  tee -a $asd_init_debug_file
lscpu 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" |  tee -a $asd_init_debug_file
lsmem 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
nvcc --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" |  tee -a $asd_init_debug_file

# Success message
cat > /tmp/success_msg.txt <<- EOM

--> Localhost URL: http://localhost:$HTTP_PORT

+----------------------------------------------------------------------------+
|             +++ The ASD Container is currently running +++                 |
+-----------------------------------------+----------------------------------+
| To stop the container run:              | docker container stop asd        |
| To delete the container run:            | docker container rm asd          |
+-----------------------------------------+----------------------------------+
EOM

# Setting up necessary variables for execution
export PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"

# Starts main ASD Python Stremlit app
cat /tmp/success_msg.txt
streamlit run /opt/asd/python-asd/src/asd/ASD-Project_Intro.py --server.port $HTTP_PORT --logger.level debug
EOFMAIN

# Build Docker Container Image
cd $asd_folder
image_downloaded=$(docker images --filter=reference='$CONTAINER_NAME*:*$CONTAINER_NAME' | wc -l)

if [ $image_downloaded -eq 1 ]; then
    docker build -t $IMAGE_NAME:$IMAGE_VERSION . || { echo 'Docker build failed' ; exit 1; }
else
    echo "+++ ASD Docker image is already available locally +++"
fi

sleep 5

# Start ASD Docker container
SUCCESS_MSG="+++ ASD Container started successfully +++"
if docker container run -d --name $CONTAINER_NAME -p $HTTP_PORT:$HTTP_PORT --hostname main-asd --privileged $IMAGE_NAME:$IMAGE_VERSION ; then
    echo -e "$SUCCESS_MSG"
else
    echo -e ""
    echo -e "!!! It seems that it was an issue executing the ASD container. Do you have Docker installed?"
    echo -e "!!! If the container is already running intentionally, please ignore this message !!!"
    echo -e "!!! If the container already exists try to run:"
    echo -e ""
    echo -e "    docker container start asd && docker container logs asd"
fi
echo -e ""
# Prints container logs
sleep 5
docker container logs asd
# Change back to User's home directory
cd $HOME