#!/bin/bash

# Configuration
REMOTE_USER="derek"
REMOTE_HOST="thelio-kim"
LOCAL_BASE="/Users/kuangda/Developer/1-projects/4-codex-analysis/data/FallopianTube"
REMOTE_BASE="/mnt/data-3/0-projects/codex-analysis/data/FallopianTube"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Annotation File Upload Script ===${NC}"
echo -e "${BLUE}Local base: ${LOCAL_BASE}${NC}"
echo -e "${BLUE}Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}${NC}"
echo ""

# Array of file paths (same as in the Python script)
declare -a file_paths=(
    "D10/img0003/img0003"
    "D11/img0005/img0005" 
    "D13/img0007/img0007"
    "D14/img0009/img0009"
    "D15/img0011/img0011"
    "D16/img0001/img0001"
    "D17/img0013/img0013"
    "D18/img0015/img0015"
)

# Function to upload files for a single image
upload_files() {
    local file_path="$1"
    local base_name=$(basename "$file_path")
    local dir_path=$(dirname "$file_path")
    
    local local_dir="${LOCAL_BASE}/${dir_path}"
    local remote_dir="${REMOTE_BASE}/${dir_path}"
    
    local json_file="${local_dir}/${base_name}_annotations.json"
    local png_file="${local_dir}/${base_name}_annotation_overview.png"
    
    echo -e "${YELLOW}Processing: ${base_name}${NC}"
    
    # Check if annotation files exist
    local files_found=0
    local files_uploaded=0
    
    if [[ -f "$json_file" ]]; then
        echo -e "  📄 Found: ${base_name}_annotations.json"
        files_found=$((files_found + 1))
    else
        echo -e "  ${RED}⚠️  Missing: ${base_name}_annotations.json${NC}"
    fi
    
    if [[ -f "$png_file" ]]; then
        echo -e "  🖼️  Found: ${base_name}_annotation_overview.png" 
        files_found=$((files_found + 1))
    else
        echo -e "  ${RED}⚠️  Missing: ${base_name}_annotation_overview.png${NC}"
    fi
    
    if [[ $files_found -eq 0 ]]; then
        echo -e "  ${RED}❌ No annotation files found, skipping${NC}"
        return 1
    fi
    
    # Create remote directory if it doesn't exist
    echo -e "  📁 Ensuring remote directory exists..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${remote_dir}'" 2>/dev/null
    
    if [[ $? -ne 0 ]]; then
        echo -e "  ${RED}❌ Failed to create remote directory${NC}"
        return 1
    fi
    
    # Upload JSON file
    if [[ -f "$json_file" ]]; then
        echo -e "  ⬆️  Uploading JSON file..."
        scp "$json_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/" 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo -e "  ${GREEN}✅ JSON file uploaded successfully${NC}"
            files_uploaded=$((files_uploaded + 1))
        else
            echo -e "  ${RED}❌ Failed to upload JSON file${NC}"
        fi
    fi
    
    # Upload PNG file  
    if [[ -f "$png_file" ]]; then
        echo -e "  ⬆️  Uploading PNG file..."
        scp "$png_file" "${REMOTE_USER}@${REMOTE_HOST}:${remote_dir}/" 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo -e "  ${GREEN}✅ PNG file uploaded successfully${NC}"
            files_uploaded=$((files_uploaded + 1))
        else
            echo -e "  ${RED}❌ Failed to upload PNG file${NC}"
        fi
    fi
    
    echo -e "  📊 Uploaded ${files_uploaded}/${files_found} files"
    echo ""
    
    return 0
}

# Function to test connection
test_connection() {
    echo -e "${BLUE}Testing connection to remote server...${NC}"
    ssh -o ConnectTimeout=10 "${REMOTE_USER}@${REMOTE_HOST}" "echo 'Connection successful'" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Connection test passed${NC}"
        return 0
    else
        echo -e "${RED}❌ Connection test failed${NC}"
        echo -e "${RED}Please check:${NC}"
        echo -e "${RED}  - Network connectivity${NC}"
        echo -e "${RED}  - SSH key authentication${NC}"
        echo -e "${RED}  - Remote server availability${NC}"
        return 1
    fi
}

# Function to show summary
show_summary() {
    local total_processed="$1"
    local total_success="$2"
    
    echo -e "${BLUE}=== Upload Summary ===${NC}"
    echo -e "📁 Total files processed: ${total_processed}"
    echo -e "${GREEN}✅ Successfully uploaded: ${total_success}${NC}"
    
    if [[ $total_success -eq $total_processed ]]; then
        echo -e "${GREEN}🎉 All annotation files uploaded successfully!${NC}"
    else
        local failed=$((total_processed - total_success))
        echo -e "${RED}⚠️  ${failed} files failed to upload${NC}"
    fi
}

# Main execution
main() {
    # Check if we're in the right directory
    if [[ ! -d "$LOCAL_BASE" ]]; then
        echo -e "${RED}❌ Local base directory not found: ${LOCAL_BASE}${NC}"
        echo -e "${RED}Please check the LOCAL_BASE path in the script${NC}"
        exit 1
    fi
    
    # Test connection first
    if ! test_connection; then
        exit 1
    fi
    
    echo ""
    echo -e "${BLUE}Starting upload process...${NC}"
    echo ""
    
    local total_processed=0
    local total_success=0
    
    # Process each file path
    for file_path in "${file_paths[@]}"; do
        if upload_files "$file_path"; then
            total_success=$((total_success + 1))
        fi
        total_processed=$((total_processed + 1))
    done
    
    echo ""
    show_summary "$total_processed" "$total_success"
}

# Command line options
case "${1:-}" in
    --dry-run)
        echo -e "${YELLOW}🔍 Dry run mode - checking local files only${NC}"
        echo ""
        for file_path in "${file_paths[@]}"; do
            base_name=$(basename "$file_path")
            dir_path=$(dirname "$file_path")
            local_dir="${LOCAL_BASE}/${dir_path}"
            
            echo -e "${YELLOW}Checking: ${base_name}${NC}"
            
            json_file="${local_dir}/${base_name}_annotations.json"
            png_file="${local_dir}/${base_name}_annotation_overview.png"
            
            [[ -f "$json_file" ]] && echo -e "  ✅ ${base_name}_annotations.json" || echo -e "  ❌ ${base_name}_annotations.json"
            [[ -f "$png_file" ]] && echo -e "  ✅ ${base_name}_annotation_overview.png" || echo -e "  ❌ ${base_name}_annotation_overview.png"
            echo ""
        done
        ;;
    --help|-h)
        echo "Upload Annotations Script"
        echo ""
        echo "Usage:"
        echo "  $0              - Upload all annotation files"
        echo "  $0 --dry-run    - Check local files without uploading"
        echo "  $0 --help       - Show this help message"
        echo ""
        echo "Configuration:"
        echo "  Local base:  $LOCAL_BASE"
        echo "  Remote:      $REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE"
        ;;
    "")
        main
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac 