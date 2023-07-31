#!/bin/bash

#! Make sure to regularly update the tailscale version and system packages

# -----------------------------------------------------------------------------
# System Hostname
# -----------------------------------------------------------------------------

sudo hostname ray-node
sudo hostnamectl set-hostname ray-node

# -----------------------------------------------------------------------------
# OS packages installation
# -----------------------------------------------------------------------------

# Allow apt update and install to run without interruption
echo "\$nrconf{restart} = 'a';" >> /etc/needrestart/needrestart.conf

# Update package lists for upgrades for packages that need upgrading, as well as new packages that have just come to the repositories
sudo apt update -y

# Remove default Python installation
sudo apt remove python3-apt -y

# Install necessary packages for building other software (build-essential, gcc, make), handling archives (zip, unzip), git for version control, curl for data transfer, and some necessary libraries
sudo apt install -y build-essential wget tar screen less jq git curl zip unzip gcc make openssl libssl-dev libffi-dev zlib1g-dev nano pciutils r-base r-base-dev libgmp-dev libmpfr-dev

# -----------------------------------------------------------------------------
# NodeJS
# -----------------------------------------------------------------------------

# Install Node.js 18.x
curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# -----------------------------------------------------------------------------
# Python 3.8.10 installation
# -----------------------------------------------------------------------------

# Download Python 3.8.10, extract and install it
cd $HOME
wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
tar xvfz Python-3.8.10.tgz
cd Python-3.8.10/
./configure --enable-optimizations
make
sudo make install

# Create symbolic links to Python 3.8 in the /usr/bin directory
sudo ln -s -f /usr/local/bin/python3.8 /usr/bin/python3
sudo ln -s -f /usr/local/bin/python3.8 /usr/bin/python

# Alias python and python3 to python3.8
echo "alias python='/usr/local/bin/python3.8'" >> ~/.bashrc
echo "alias python3='/usr/local/bin/python3.8'" >> ~/.bashrc
source ~/.bashrc

# Install pip for Python 3.8
wget https://bootstrap.pypa.io/get-pip.py
/usr/local/bin/python3.8 get-pip.py

# -----------------------------------------------------------------------------
# AWS CLI and SSM Agent
# -----------------------------------------------------------------------------

# Install AWS CLI
cd $HOME
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
export aws_region=$(curl http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')
echo '''export aws_region=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')''' >> $HOME/.bashrc
sudo echo "aws_region=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')" >> /etc/environment

# Prerequisites for SSM Agent
/usr/local/bin/python3.8 -m pip install easy_install
/usr/local/bin/python3.8 -m easy_install https://s3.amazonaws.com/cloudformation-examples/aws-cfn-bootstrap-py3-latest.tar.gz

# Activate SSM Agent
sudo snap install amazon-ssm-agent --classic
sudo snap start amazon-ssm-agent

# -----------------------------------------------------------------------------
# Tailscale client installation
# -----------------------------------------------------------------------------

cd $HOME
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list | tee /etc/apt/sources.list.d/tailscale.list  
sudo apt update
sudo apt install tailscale -y

# -----------------------------------------------------------------------------
# Ray and Ray Dashboard
# -----------------------------------------------------------------------------

# Clone the Ray repository and install Bazel
cd $HOME
git clone https://github.com/ray-project/ray || true
ray/ci/env/install-bazel.sh

# Build the Ray dashboard
cd ray/python/ray/dashboard/client
npm ci
npm run build

# Install Python dependencies for Ray
/usr/local/bin/python3.8 -m pip install boto3>=1.4.8 cython==0.29.32 aiohttp grpcio psutil setproctitle

# Install Ray
cd ~/ray/python
/usr/local/bin/python3.8 -m pip install -e . --verbose

# -----------------------------------------------------------------------------
# ASD Installation and configuration
# -----------------------------------------------------------------------------

# Creates asd directory if not exist
sudo mkdir -p /opt/asd

# Python dependencies installation
/usr/local/bin/python3.8 -m pip install Cython
/usr/local/bin/python3.8 -m pip install GenDoc -U
/usr/local/bin/python3.8 -m pip install git+https://github.com/h2oai/datatable.git
/usr/local/bin/python3.8 -m pip install bokeh==2.4.3

# ASD Python requirements
cat << EOF > $HOME/requirements.txt
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
tensorflow==2.13.0
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
/usr/local/bin/python3.8 -m pip install -r $HOME/requirements.txt

/usr/local/bin/python3.8 -m pip freeze > /opt/asd/asd-requirements-with-versions.txt

# Setting Python PATH
echo 'PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"' >> /etc/environment

# Protobuf settings
curl -o /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/builder.py https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py

# Installation of R dependencies
Rscript -e 'install.packages(c("base", "base64enc", "bit", "bit64", "boot", "class", "cluster", "codetools", "compiler", "CVXR", "datasets", "dbscan", "digest", "ECOSolveR", "evaluate", "fastcluster", "fastmap", "foreign", "glue", "gmp", "graphics", "grDevices", "grid", "highr", "htmltools", "htmlwidgets", "jsonlite", "KernSmooth", "knitr", "labdsv", "lattice", "magrittr", "maotai", "MASS", "Matrix", "mclustcomp", "methods", "mgcv", "minpack.lm", "nlme", "nnet", "osqp", "parallel", "R6", "RANN", "rbibutils", "Rcpp", "RcppArmadillo", "RcppDE", "RcppDist", "RcppEigen", "Rcsdp", "Rdimtools", "rdist", "Rdpack", "rgl", "rlang", "Rmpfr", "rpart", "RSpectra", "Rtsne", "scatterplot3d", "scs", "shapes", "spatial", "splines", "stats", "stats4", "stringi", "stringr", "survival", "tcltk", "tools", "utils", "xfun", "yaml"), repos="https://cloud.r-project.org")' | tee -a /opt/asd/$CONTAINER_NAME-r-dependencies-installation-log.txt

# Clone the ASD git repository (excluding the 'container folder')
sudo git clone https://github.com/limacarvalho/automatedscientificdiscovery.git /opt/asd/python-asd
sudo chown -R ubuntu:ubuntu /opt/asd/
cd /opt/asd/python-asd

# Setting up necessary variables for execution of ASD app (if needed)
echo 'export PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"' >> $HOME/.bashrc
export PYTHONPATH="/opt/asd/python-asd/src/asd:/opt/asd/python-asd/src/asd/complexity:/opt/asd/python-asd/src/asd/complexity/dim_reduce:/opt/asd/python-asd/src/asd/predictability:/opt/asd/python-asd/src/asd/relevance:/opt/asd/python-asd/src/asd/relevance/ml:/opt/asd/python-asd/src/asd/relevance/utils"


# Starts main ASD Python Stremlit app Message
cat << EOF >> $HOME/.bashrc
echo -e "
+------------------------------------------------------------------------------------------------------+
|                    +++ The ASD Web App can be started manually:  Info  +++                           |
+------------------------------------------------------+-----------------------------------------------+
| Execute the following command:                                                                       |
| streamlit run /opt/asd/python-asd/src/asd/ASD-Project_Intro.py --server.port 80 --logger.level debug |
+------------------------------------------------------+-----------------------------------------------+
"
EOF

# Forces bash update
source $HOME/.bashrc

# -----------------------------------------------------------------------------
# Tailscale connect service configuration
# -----------------------------------------------------------------------------

# Create Tailscale connect script file used by systemd
tailscale_connect_service_path='/opt/asd/tailscale_connect_service.sh'
sudo echo "#!/bin/bash" >> $tailscale_connect_service_path
sudo echo -e "" >> $tailscale_connect_service_path
sudo echo "aws_lightsail_headscale_svc_name=headscale-asd" >> $tailscale_connect_service_path
sudo echo "aws_lightsail_headscale_container_name=headscale" >> $tailscale_connect_service_path
sudo echo "tailscale_hostname=ray-node" >> $tailscale_connect_service_path
sudo echo "aws_region=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')" >> $tailscale_connect_service_path

cat << 'EOF' >> $tailscale_connect_service_path

# Prevent Multiple Instances. Implements locking mechanism to prevent multiple instances from running
lockfile="/tmp/mylockfile"
if [ -e ${lockfile} ] && kill -0 `cat ${lockfile}`; then
    echo "+++ already running +++"
    exit
fi

trap "rm -f ${lockfile}; exit" INT TERM EXIT
echo $$ > ${lockfile}

# Main Tailscale connect function
tailscale() {
    cd "$HOME"

    # Main logic for Tailscale connection to Headscale sever
    while true; do
        # Attemp to get the headscale URL, Auth Key and Deployment Status
        echo -e "+++ Trying to retrieve the AWS Lightsail HTTPS URL and pre-generated auth key for the Headscale container/service..."
        headscale_url=$(aws lightsail get-container-services --service-name $aws_lightsail_headscale_svc_name --region $aws_region | jq -r '.containerServices | .[] | .url')
        headscale_auth_key=$(aws lightsail get-container-log --service-name $aws_lightsail_headscale_svc_name --container-name $aws_lightsail_headscale_container_name --filter-pattern "VAR_NODES_AUTH_KEY" --region $aws_region | jq -r '.logEvents[-1].message' | sed -r 's/^VAR_NODES_AUTH_KEY: //')
        headscale_deployment_status=$(aws lightsail get-container-service-deployments --service-name $aws_lightsail_headscale_svc_name --region $aws_region | jq -r '.deployments | .[] | .state')

        # Check if the $headscale_url variable's value contains "https://", if $headscale_auth_key contains the auth key and if Deployment is 'ACTIVE'. This because the command always returns a 0 exit code
        if [[ $headscale_url == https://* ]] && [[ -n $headscale_auth_key ]] && [[ $headscale_auth_key != "null" ]] && [[ $headscale_deployment_status == "ACTIVE" ]]; then
            echo -e ""
            echo -e "+++ The Headscale URL is: ${headscale_url} +++"
            echo -e "+++ The Headscale Auth Key is: ${headscale_auth_key} +++"
            echo -e "+++ The Headscale Deployment Status = ${headscale_deployment_status} +++"
            echo -e ""
            break
        fi

        # Wait 15 seconds before retrying
        sleep 15
    done

    # Initialize connection from local tailscale to remote Headscale server on AWS
    /usr/sbin/tailscaled --cleanup
    screen -dmS tailscale_state bash -c "/usr/sbin/tailscaled --state=/var/lib/tailscale/tailscaled.state --socket=/run/tailscale/tailscaled.sock --port=0"
    screen -dmS tailscale bash -c "tailscale up --auth-key $headscale_auth_key --hostname $tailscale_hostname --login-server $headscale_url --accept-dns=true"

    sleep 10
    export tailscale_node_ip=$(/usr/bin/tailscale ip)
    echo -e "-----------------------TAILSCALE INFO ----------------------------"
    echo -e "TAILSCALE_IPs: \n$tailscale_node_ip"
    echo -e "------------------------------------------------------------------"

}

tailscale
EOF

# Make script executable
chmod +x $tailscale_connect_service_path

# Create Tailscale connect systemd config file
cat << 'EOF' > /etc/systemd/system/tailscale-connect.service
[Unit]
Description=Tailscale Connect

[Service]
Type=simple
ExecStart=/opt/asd/tailscale_connect_service.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Create Tailscale connect systemd timer
cat << 'EOF' > /etc/systemd/system/tailscale-connect.timer
[Unit]
Description=Runs Tailscale Connect script every 10 minutes

[Timer]
OnCalendar=*:0/10
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable and start the timer
sudo systemctl enable tailscale-connect.service
sudo systemctl start tailscale-connect.service
sudo systemctl enable tailscale-connect.timer
sudo systemctl start tailscale-connect.timer