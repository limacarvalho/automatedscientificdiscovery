#!bin/bash

# +------------------------------+----------------------------------+
# |                 ASD Container runtime script                    |
# +------------------------------+----------------------------------+
# | Version                      | 1.1                              |
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

--> Localhost URL: http://localhost:80

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
streamlit run /opt/asd/python-asd/src/asd/ASD-Project_Intro.py --server.port 80 --logger.level debug
