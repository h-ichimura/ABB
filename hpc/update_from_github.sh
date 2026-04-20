#!/bin/bash
# Update ABB code on Puma from GitHub.
# Run this on Puma: bash ~/ABB/hpc/update_from_github.sh
#
# SAFE UPDATE: Never deletes results or logs. Old code is moved to
# ~/ABB_backup_TIMESTAMP, not deleted.
#
# Usage:
#   1. Ask Claude to run: gh repo edit h-ichimura/ABB --visibility public
#   2. On Puma: bash ~/ABB/hpc/update_from_github.sh
#   3. Ask Claude to run: gh repo edit h-ichimura/ABB --visibility private

set -e  # Exit on any error

cd ~
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
rm -rf ABB_latest ABB-master
unzip -q ABB_latest.zip
mv ABB-master ABB_latest

# Copy results and logs INTO the new code (use cp, not mv, so originals are safe)
for DIR in results logs hpc/results hpc/logs; do
    if [ -d ~/ABB/$DIR ]; then
        COUNT=$(ls ~/ABB/$DIR/ 2>/dev/null | wc -l)
        echo "Copying existing $DIR/ ($COUNT files)..."
        mkdir -p ABB_latest/$DIR
        cp -a ~/ABB/$DIR/. ABB_latest/$DIR/
    fi
done

# Verify results were copied
OLD_COUNT=$(ls ~/ABB/hpc/results/ 2>/dev/null | wc -l)
NEW_COUNT=$(ls ABB_latest/hpc/results/ 2>/dev/null | wc -l)
if [ "$OLD_COUNT" -gt 0 ] && [ "$NEW_COUNT" -lt "$OLD_COUNT" ]; then
    echo "ERROR: Result copy failed! Old=$OLD_COUNT, New=$NEW_COUNT"
    echo "Aborting. Old ABB/ is untouched."
    rm -rf ABB_latest ABB_latest.zip
    exit 1
fi

# Move old ABB to backup (NEVER delete)
if [ -d ~/ABB ]; then
    BACKUP="ABB_backup_${TIMESTAMP}"
    echo "Moving old ABB/ to ~/$BACKUP/"
    mv ~/ABB ~/$BACKUP
fi

# Install new code
mv ABB_latest ~/ABB

# Create directories if needed
mkdir -p ~/ABB/results ~/ABB/logs ~/ABB/hpc/results ~/ABB/hpc/logs

# Cleanup zip only
rm -f ABB_latest.zip

echo ""
echo "Updated successfully."
echo "Files: $(find ~/ABB -name '*.jl' | wc -l) Julia, $(find ~/ABB -name '*.sh' | wc -l) shell"
echo "Results: $(ls ~/ABB/hpc/results/ 2>/dev/null | wc -l) files (copied from old)"
echo "Old code backed up to: ~/$BACKUP/"
echo ""
echo "To clean up old backups later: ls -d ~/ABB_backup_*"
