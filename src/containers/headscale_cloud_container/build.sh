#!/bin/bash

#! Make sure to regularly update the headscale version and system packages

apt update -y -qq > /dev/null

# Install necessary tools quietly and without interaction
apt install nano zip curl wget tar screen less nginx jq -y -qq > /dev/null

# Move to home directory
cd "$HOME"

# Download headscale package from GitHub
wget https://github.com/juanfont/headscale/releases/download/v0.22.3/headscale_0.22.3_linux_amd64.deb

# Install the downloaded package
dpkg -i headscale_0.22.3_linux_amd64.deb

# Create an nginx configuration file to act as a proxy to headscale service
cat > /etc/nginx/sites-available/default << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Restart nginx service to apply new configurations
service nginx restart
