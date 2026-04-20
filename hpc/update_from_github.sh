#!/bin/bash
# Update ABB code on Puma from GitHub.
# Run this on Puma: bash ~/ABB/hpc/update_from_github.sh
#
# This downloads the latest code from GitHub without needing
# git authentication. Requires Claude to make the repo public
# briefly before running.
#
# Usage:
#   1. Ask Claude to run: gh repo edit h-ichimura/ABB --visibility public
#   2. On Puma: bash ~/ABB/hpc/update_from_github.sh
#   3. Ask Claude to run: gh repo edit h-ichimura/ABB --visibility private

cd ~

# Download latest
echo "Downloading latest code from GitHub..."
rm -f ABB_latest.zip
curl -sL https://github.com/h-ichimura/ABB/archive/refs/heads/master.zip -o ABB_latest.zip

# Check download succeeded (should be > 1KB)
SIZE=$(stat -c%s ABB_latest.zip 2>/dev/null || stat -f%z ABB_latest.zip 2>/dev/null)
if [ "$SIZE" -lt 1000 ]; then
    echo "ERROR: Download failed (${SIZE} bytes). Is the repo public?"
    echo "Ask Claude to run: gh repo edit h-ichimura/ABB --visibility public"
    rm -f ABB_latest.zip
    exit 1
fi

# Extract to temp dir
rm -rf ABB_latest
unzip -q ABB_latest.zip
mv ABB-master ABB_latest

# Preserve results and logs from current installation
if [ -d ~/ABB/results ]; then
    echo "Preserving existing results/ directory..."
    mv ~/ABB/results ABB_latest/results
fi
if [ -d ~/ABB/logs ]; then
    echo "Preserving existing logs/ directory..."
    mv ~/ABB/logs ABB_latest/logs
fi
if [ -d ~/ABB/hpc/results ]; then
    echo "Preserving existing hpc/results/ directory..."
    mkdir -p ABB_latest/hpc
    mv ~/ABB/hpc/results ABB_latest/hpc/results
fi
if [ -d ~/ABB/hpc/logs ]; then
    echo "Preserving existing hpc/logs/ directory..."
    mkdir -p ABB_latest/hpc
    mv ~/ABB/hpc/logs ABB_latest/hpc/logs
fi

# Replace old with new
rm -rf ~/ABB
mv ABB_latest ~/ABB

# Create directories if needed
mkdir -p ~/ABB/results ~/ABB/logs ~/ABB/hpc/results ~/ABB/hpc/logs

# Cleanup
rm -f ABB_latest.zip

echo "Updated successfully."
echo "Files: $(find ~/ABB -name '*.jl' | wc -l) Julia, $(find ~/ABB -name '*.sh' | wc -l) shell"
echo "Results preserved: $(ls ~/ABB/results/ 2>/dev/null | wc -l) files"
