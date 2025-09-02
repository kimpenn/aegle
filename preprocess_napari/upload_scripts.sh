#!/bin/bash

# Configuration
REMOTE_USER="derek"
REMOTE_HOST="thelio-kim"
LOCAL_DIR="/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/preprocess_napari"
REMOTE_DIR="/mnt/data-3/0-projects/codex-analysis/0-phenocycler-penntmc-pipeline/preprocess_napari"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Script Upload Tool ===${NC}"
echo -e "${BLUE}Local dir: ${LOCAL_DIR}${NC}"
echo -e "${BLUE}Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}${NC}"
echo ""

# Files to upload
FILES=(
    "napari_annotation.py"
    "README.md"
    "upload_annotations.sh"
    "upload_scripts.sh"
)

# Function to test connection
test_connection() {
    echo -e "${YELLOW}Testing connection to ${REMOTE_HOST}...${NC}"
    if ssh -o ConnectTimeout=10 -o BatchMode=yes ${REMOTE_USER}@${REMOTE_HOST} "echo 'Connection successful'" 2>/dev/null; then
        echo -e "${GREEN}✓ Connection successful${NC}"
        return 0
    else
        echo -e "${RED}✗ Connection failed${NC}"
        echo -e "${RED}Please check your SSH connection to ${REMOTE_USER}@${REMOTE_HOST}${NC}"
        return 1
    fi
}

# Function to create remote directory if it doesn't exist
create_remote_dir() {
    echo -e "${YELLOW}Ensuring remote directory exists...${NC}"
    if ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}" 2>/dev/null; then
        echo -e "${GREEN}✓ Remote directory ready${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to create remote directory${NC}"
        return 1
    fi
}

# Function to upload a file
upload_file() {
    local file=$1
    local local_path="${LOCAL_DIR}/${file}"
    local remote_path="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${file}"
    
    echo -e "${YELLOW}Uploading ${file}...${NC}"
    
    # Check if local file exists
    if [[ ! -f "$local_path" ]]; then
        echo -e "${RED}✗ Local file not found: ${local_path}${NC}"
        return 1
    fi
    
    # Upload file
    if scp "$local_path" "$remote_path" 2>/dev/null; then
        echo -e "${GREEN}✓ Successfully uploaded ${file}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to upload ${file}${NC}"
        return 1
    fi
}

# Main execution
main() {
    # Check for dry-run flag
    if [[ "$1" == "--dry-run" ]]; then
        echo -e "${YELLOW}DRY RUN MODE - No files will be uploaded${NC}"
        echo ""
        
        for file in "${FILES[@]}"; do
            local_path="${LOCAL_DIR}/${file}"
            if [[ -f "$local_path" ]]; then
                echo -e "${GREEN}✓ Would upload: ${file}${NC}"
            else
                echo -e "${RED}✗ Missing file: ${file}${NC}"
            fi
        done
        
        echo ""
        echo -e "${BLUE}To actually upload, run: $0${NC}"
        return 0
    fi
    
    # Check for help flag
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --dry-run    Show what would be uploaded without actually uploading"
        echo "  --help, -h   Show this help message"
        echo ""
        echo "This script uploads the following files:"
        for file in "${FILES[@]}"; do
            echo "  - ${file}"
        done
        return 0
    fi
    
    # Test connection
    if ! test_connection; then
        exit 1
    fi
    
    echo ""
    
    # Create remote directory
    if ! create_remote_dir; then
        exit 1
    fi
    
    echo ""
    
    # Upload files
    local success_count=0
    local total_count=${#FILES[@]}
    
    for file in "${FILES[@]}"; do
        if upload_file "$file"; then
            ((success_count++))
        fi
        echo ""
    done
    
    # Summary
    echo -e "${BLUE}=== Upload Summary ===${NC}"
    echo -e "${GREEN}Successfully uploaded: ${success_count}/${total_count} files${NC}"
    
    if [[ $success_count -eq $total_count ]]; then
        echo -e "${GREEN}✓ All files uploaded successfully!${NC}"
        exit 0
    else
        echo -e "${RED}✗ Some files failed to upload${NC}"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
