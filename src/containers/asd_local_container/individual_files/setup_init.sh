#!/bin/bash

# +------------------------------+----------------------------------+
# |              ASD Container runtime setup script                 |
# +------------------------------+----------------------------------+
# | Version                      | 1.0                              |
# | Language                     | Linux Bash                       |
# | Platform                     | x86_64                           |
# | Input Parameters             | None                             |
# | GPU / non-GPU support        | Yes / Yes                        |
# | Runs on Docker Image         | tensorflow/tensorflow:2.11.0-gpu |
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
nvidia-smi 2> /dev/null | tee -a $asd_init_debug_file
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
gpustat
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
ray[default]
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

# Tensorflow settings
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
echo 'LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/:/usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"' >> /etc/environment

# Setting Python PATH
echo 'PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"' >> /etc/environment

# Protobuf settings
curl -o /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/builder.py https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py

# CUDA Numa Node settings (disabled by default)
#cat /sys/bus/pci/devices/0000\:00\:1e.0/numa_node # if value != 0 then sudo echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:00\:1e.0/numa_node in host machine
#TODO "I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA"

# Installation of R dependencies
Rscript -e 'install.packages(c("base", "base64enc", "bit", "bit64", "boot", "class", "cluster", "codetools", "compiler", "CVXR", "datasets", "dbscan", "digest", "ECOSolveR", "evaluate", "fastcluster", "fastmap", "foreign", "glue", "gmp", "graphics", "grDevices", "grid", "highr", "htmltools", "htmlwidgets", "jsonlite", "KernSmooth", "knitr", "labdsv", "lattice", "magrittr", "maotai", "MASS", "Matrix", "mclustcomp", "methods", "mgcv", "minpack.lm", "nlme", "nnet", "osqp", "parallel", "R6", "RANN", "rbibutils", "Rcpp", "RcppArmadillo", "RcppDE", "RcppDist", "RcppEigen", "Rcsdp", "Rdimtools", "rdist", "Rdpack", "rgl", "rlang", "Rmpfr", "rpart", "RSpectra", "Rtsne", "scatterplot3d", "scs", "shapes", "spatial", "splines", "stats", "stats4", "stringi", "stringr", "survival", "tcltk", "tools", "utils", "xfun", "yaml"), repos="https://cloud.r-project.org")' | tee -a /opt/asd/asd-container-r-dependencies-installation-log.txt

# Clone the ASD git repository (excluding the 'container folder')
git clone --depth 1 --filter=blob:none --sparse https://gitlab+deploy-token-1651733:jUZKE9xQWjsFxS3yU-2s@gitlab.com/automatedscientificdiscovery/python-asd.git /opt/asd/python-asd
cd /opt/asd/python-asd
git sparse-checkout init --cone
git sparse-checkout set --no-cone '/*' '!src/containers/asd_local_container'
# git sparse-checkout set src (Only clones the src/ folder and all content in it, ignoring all the remaining folders. Files on the top directory are cloned)
# git sparse-checkout set --no-cone '/*' '!src/asd/datasets' '!src/asd/anyotherfolder' (Clones all the repoitory except for the folders specified)
# More info about git sparse checkout at https://git-scm.com/docs/git-sparse-checkout

exit 0