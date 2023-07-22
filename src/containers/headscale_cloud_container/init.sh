#!/bin/bash

# Restart nginx service to apply new configurations
service nginx restart

# Retrieve the public IP address of the machine using ifconfig.me service
export public_ipv4=$(curl ifconfig.me)

# Create a configuration file for headscale with specified settings
cat << EOF > /etc/headscale/config.yaml
---
server_url: http://127.0.0.1:8080
listen_addr: 0.0.0.0:8080
metrics_listen_addr: 127.0.0.1:9090
grpc_listen_addr: 127.0.0.1:50443
grpc_allow_insecure: false
private_key_path: /var/lib/headscale/private.key
noise:
  private_key_path: /var/lib/headscale/noise_private.key
ip_prefixes:
  - fd7a:115c:a1e0::/48
  - 100.64.0.0/10
derp:
  server:
    enabled: false
    region_id: 999
    region_code: "headscale"
    region_name: "Headscale Embedded DERP"
    stun_listen_addr: "0.0.0.0:3478"
  urls:
    - https://controlplane.tailscale.com/derpmap/default
  paths: []
  auto_update_enabled: true
  update_frequency: 24h
disable_check_updates: false
ephemeral_node_inactivity_timeout: 30m
node_update_check_interval: 10s
db_type: sqlite3
db_path: /var/lib/headscale/db.sqlite
acme_url: https://acme-v02.api.letsencrypt.org/directory
acme_email: ""
tls_letsencrypt_hostname: ""
tls_letsencrypt_cache_dir: /var/lib/headscale/cache
tls_letsencrypt_challenge_type: HTTP-01
tls_letsencrypt_listen: ":http"
tls_cert_path: ""
tls_key_path: ""
log:
  format: text
  level: info
acl_policy_path: ""
dns_config:
  override_local_dns: true
  nameservers:
    - 1.1.1.1
  domains: []
  magic_dns: true
  base_domain: example.com
unix_socket: /var/run/headscale/headscale.sock
unix_socket_permission: "0770"
logtail:
  enabled: false
randomize_client_port: false
EOF

# Start headscale service in a detached screen and log its output to a file
screen -dmS headscale bash -c "/usr/bin/headscale serve | tee -a /tmp/headscale_logfile.txt"

# Create a user named "asd" in headscale
headscale users create asd

# Create a reusable pre-authorization key for user "asd"
export headscale_preauthkey=$(headscale preauthkeys create --reusable --user asd)

# Print out important variables
echo -e "-----------------------HEADSCALE VARIABLES----------------------------"
echo -e "VAR_IPV4: $public_ipv4"
echo -e "VAR_NODES_AUTH_KEY: $headscale_preauthkey"
echo -e "----------------------------------------------------------------------"

# Sleep for 1 minute
sleep 60

# Endless loop to periodically retrieve and print out registered nodes
while true
do
    # Get current time in ISO 8601 format
    current_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo -e "-------------------RETRIEVING REGISTERED NODES------------------------"
    echo -e "TIMESTAMP: $current_time"

    # Get the list of registered nodes in JSON format and format the output
    headscale nodes list --output json | jq -r 'if type=="array" then .[] | "NODE_NAME: " + .given_name + "\nIP_ADDRESS:\n" + .ip_addresses[] else empty end'
    echo -e ""

    # Sleep for 900 seconds (15 minutes) before the next iteration
    sleep 900
done
