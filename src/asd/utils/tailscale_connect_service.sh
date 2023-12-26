#!/bin/bash

aws_lightsail_headscale_svc_name=headscale-asd
aws_lightsail_headscale_container_name=headscale
asd_container_uuid_file=/root/.asd_container_uuid
tailscale_hostname=ray-node
aws_region=us-east-1

# Prevent Multiple Instances. Implements locking mechanism to prevent multiple instances from running
lockfile="/tmp/mylockfile"
if [ -e ${lockfile} ] && kill -0 `cat ${lockfile}`; then
    echo "+++ already running +++"
    exit
fi

# Retrieve AsdDeploymentCount
AsdDeploymentCount=$(curl -s http://169.254.169.254/latest/meta-data/iam/info | jq -r '.InstanceProfileArn' 2> /dev/null | sed 's/.*asg-asd-\([0-9]\+\)-InstanceProfile.*/\1/')

# Validate AsdDeploymentCount variable and build string for aws_lightsail_headscale_svc_name variable
if [[ $AsdDeploymentCount =~ ^[1-9][0-9]{0,2}$ ]]; then
    # If AsdDeploymentCount variable's value is an integer from 1 to 999, build aws_lightsail_headscale_svc_name
    aws_lightsail_headscale_svc_name="headscale-asd-${AsdDeploymentCount}"
else
    # If AsdDeploymentCount is not valid/present or if it cannot be retrieved from the ASD container .asd_container_uuid file, use the positional argument
    if [ -z "$1" ]; then
        AsdDeploymentCount=$(jq -r '.asd_deployment_number' $asd_container_uuid_file)
        aws_lightsail_headscale_svc_name="headscale-asd-${AsdDeploymentCount}"
    else
        aws_lightsail_headscale_svc_name=$1
    fi
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
