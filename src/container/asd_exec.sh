#!bin/bash

# +------------------------------+----------------------------------+
# |                 ASD Container runtime script                    |
# +------------------------------+----------------------------------+
# | Version                      | 1.0                              |
# | Language                     | Linux Bash                       |
# | Platform                     | x86_64                           |
# | Input Parameters             | None                             |
# | GPU / non-GPU support        | Yes / Yes                        |
# | Runs on Docker Image         | tensorflow/tensorflow:2.11.0-gpu |
# +------------------------------+----------------------------------+

# Debug information
asd_init_debug_file="/opt/asd/asd-container-start-versions.txt"

echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
echo -e Execution time: $(date -u) | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
python --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
gcc --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
nvidia-smi 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" | tee -a $asd_init_debug_file
nvcc --version 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" |  tee -a $asd_init_debug_file
lscpu 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" |  tee -a $asd_init_debug_file
lsmem 2> /dev/null | tee -a $asd_init_debug_file
echo -e "\n===============================================\n" |  tee -a $asd_init_debug_file

# Starts main AS Python Stremlit app
streamlit run /opt/asd/python-asd/src/asd/ASD-Project_Intro.py --server.port 80
