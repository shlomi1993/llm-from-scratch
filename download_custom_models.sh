#!/bin/bash

# Script to download all custom models from Google Drive
# Usage: ./download_custom_models.sh [target_directory]
# Default target: models/

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Target directory (default: models/)
TARGET_DIR="${1:-models}"

# Model information arrays
MODEL_NAMES=("foundation" "classifier" "assistant" "coder")
MODEL_FILE_IDS=("1AZP-AEEm8NJF4wGXkxEF-hxAGBAdShIt" "1ecl-LeMq3fgNBEDjkKzisWp4ube40Jb0" "1qiM0YHdnciGoUaJadnQzL7hDhiLHJ-QZ" "1ugMFBx8cfDz_q6RJ7G5YHrJhHSLTPFi7")
MODEL_FILENAMES=("foundation.zip" "spam_classifier.zip" "assistant.zip" "coder.zip")
MODEL_SUBDIRS=("pretrained" "classifier" "assistant" "coder")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Custom Models Downloader${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Target directory: ${TARGET_DIR}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to download file from Google Drive using gdown
download_with_gdown() {
    local file_id=$1
    local output=$2
    echo -e "${YELLOW}Downloading using gdown...${NC}"
    gdown "https://drive.google.com/uc?id=${file_id}" -O "${output}"
}

# Function to download file from Google Drive using wget
download_with_wget() {
    local file_id=$1
    local output=$2
    echo -e "${YELLOW}Downloading using wget...${NC}"
    # First request to get the confirmation token
    wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=${file_id}" -O- | \
        sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > /tmp/confirm.txt

    local confirm=$(cat /tmp/confirm.txt)
    if [ -z "$confirm" ]; then
        # No confirmation needed, download directly
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=${file_id}" -O "${output}"
    else
        # Use confirmation token
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" -O "${output}"
    fi
    rm -rf /tmp/cookies.txt /tmp/confirm.txt
}

# Function to download file from Google Drive using curl
download_with_curl() {
    local file_id=$1
    local output=$2
    echo -e "${YELLOW}Downloading using curl...${NC}"

    # Get the confirmation token
    curl -c /tmp/cookies.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" > /tmp/intermezzo.html

    # Extract confirm code if present
    local confirm=$(cat /tmp/intermezzo.html | grep -o 'confirm=[^&]*' | sed 's/confirm=//' | head -n 1)

    if [ -z "$confirm" ]; then
        # Try alternative: download directly with uc?id= format
        curl -L "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t" -o "${output}"
    else
        # Use the confirmation token
        curl -L -b /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" -o "${output}"
    fi

    rm -rf /tmp/cookies.txt /tmp/intermezzo.html
}

# Determine which download method to use
if command_exists gdown; then
    DOWNLOAD_METHOD="gdown"
    echo -e "${GREEN}✓ Using gdown for downloads (recommended for Google Drive)${NC}"
elif command_exists wget; then
    DOWNLOAD_METHOD="wget"
    echo -e "${GREEN}✓ Using wget for downloads${NC}"
    echo -e "${YELLOW}⚠ Note: For best results, install gdown: pip install gdown${NC}"
elif command_exists curl; then
    DOWNLOAD_METHOD="curl"
    echo -e "${GREEN}✓ Using curl for downloads${NC}"
    echo -e "${YELLOW}⚠ Note: For best results, install gdown: pip install gdown${NC}"
else
    echo -e "${RED}Error: No suitable download tool found!${NC}"
    echo -e "${YELLOW}Please install one of: gdown (pip install gdown), wget, or curl${NC}"
    exit 1
fi

# Check if unzip is available
if ! command_exists unzip; then
    echo -e "${RED}Error: unzip command not found!${NC}"
    echo -e "${YELLOW}Please install unzip to extract the downloaded files${NC}"
    exit 1
fi

echo ""

# Download and extract all models
for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    file_id="${MODEL_FILE_IDS[$idx]}"
    filename="${MODEL_FILENAMES[$idx]}"
    subdir="${MODEL_SUBDIRS[$idx]}"
    target_dir="${TARGET_DIR}/${subdir}"

    echo -e "${GREEN}Downloading: ${model_name}${NC}"

    # Create target directory if it doesn't exist
    mkdir -p "${target_dir}"

    # Download the file
    echo -e "${YELLOW}Downloading ${filename}...${NC}"
    case $DOWNLOAD_METHOD in
        gdown)
            download_with_gdown "$file_id" "$filename"
            ;;
        wget)
            download_with_wget "$file_id" "$filename"
            ;;
        curl)
            download_with_curl "$file_id" "$filename"
            ;;
    esac

    if [ $? -eq 0 ]; then
        # Check if the downloaded file is actually a zip file (not an HTML error page)
        file_type=$(file -b "${filename}" | head -c 3)
        if [[ "$file_type" == "Zip" ]]; then
            echo -e "${GREEN}✓ Download completed${NC}"
        else
            echo -e "${RED}✗ Download failed - received HTML instead of zip file${NC}"
            echo -e "${YELLOW}The file is too large for simple download. Please install gdown:${NC}"
            echo -e "${YELLOW}  pip install gdown${NC}"
            echo -e "${YELLOW}Then run this script again.${NC}"
            rm -f "${filename}"
            continue
        fi
    else
        echo -e "${RED}✗ Download failed for ${model_name}${NC}"
        continue
    fi

    # Extract the zip file
    echo -e "${YELLOW}Extracting to ${target_dir}...${NC}"
    unzip -q -o "$filename" -d "${target_dir}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Extraction completed${NC}"
        # Remove the zip file
        rm "$filename"
        echo -e "${GREEN}✓ Cleaned up ${filename}${NC}"
    else
        echo -e "${RED}✗ Extraction failed for ${filename}${NC}"
    fi

    echo ""
done

echo -e "${GREEN}All Models Downloaded!${NC}"