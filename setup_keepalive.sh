#!/bin/bash
# OCI AlwaysFree Keepalive Setup
# This script installs stress-ng and configures a systemd service to maintain 40% CPU and memory utilization.
# It runs with the lowest possible system priority so it immediately yields resources when your web app needs them.

# 1. Install stress-ng if not present
echo "Checking for stress-ng..."
if ! command -v stress-ng &> /dev/null; then
    echo "Installing stress-ng..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y stress-ng
    elif command -v yum &> /dev/null; then
        sudo yum install -y epel-release
        sudo yum install -y stress-ng
    else
        echo "Error: Package manager not supported. Install stress-ng manually."
        exit 1
    fi
else
    echo "stress-ng is already installed!"
fi

# 2. Create the systemd service file
echo "Creating /etc/systemd/system/oci-keepalive.service..."
sudo bash -c 'cat > /etc/systemd/system/oci-keepalive.service << "EOF"
[Unit]
Description=OCI AlwaysFree Idle Keepalive (Anti-Reclamation)
After=network.target

[Service]
Type=simple
# --cpu 0: Spawn 1 worker per CPU core
# --cpu-load 40: Target 40% CPU utilization
# --vm 1: Spawn 1 memory worker
# --vm-bytes 40%: Allocate 40% of available memory
# --vm-keep: Hold the memory instead of constantly reallocating (prevents memory bus thrashing)
ExecStart=/usr/bin/stress-ng --cpu 0 --cpu-load 40 --vm 1 --vm-bytes 40% --vm-keep
Restart=always
RestartSec=60

# --- Yielding Priority --- 
# Lowest priority possible so it never blocks Resumatch_OCI processing!
Nice=19
CPUSchedulingPolicy=idle
IOSchedulingClass=idle

[Install]
WantedBy=multi-user.target
EOF'

# 3. Reload systemd and start exactly as configured
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling oci-keepalive service to start on boot..."
sudo systemctl enable oci-keepalive

echo "Starting oci-keepalive service..."
sudo systemctl restart oci-keepalive

echo "Setup complete! Checking status:"
sudo systemctl status oci-keepalive --no-pager
