#!/usr/bin/env python3
"""
Compress HTML reports by resizing and converting embedded images to JPEG.

This script processes existing pipeline HTML reports to reduce file size
by converting embedded base64 PNG images to compressed JPEG format.

Usage:
    # Single file
    python scripts/compress_html_reports.py /path/to/pipeline_report.html

    # Directory (batch)
    python scripts/compress_html_reports.py /path/to/out/main/main_ft_hb/ --recursive

    # Options
    --max-width 1200     # Max image width (default: 1200)
    --quality 85         # JPEG quality 1-100 (default: 85)
    --in-place           # Overwrite originals (no .raw.html backup)
    --dry-run            # Show expected size reduction without modifying files

By default, the script renames the original to .raw.html and saves the
compressed version as the original filename (e.g., pipeline_report.html).
"""

import argparse
import base64
import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Regex pattern to match base64-encoded images in HTML
# Matches: data:image/png;base64,... or data:image/jpeg;base64,...
IMAGE_PATTERN = re.compile(
    r'data:image/(png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)',
    re.IGNORECASE
)


def compress_image(
    image_data: bytes,
    max_width: int = 1200,
    jpeg_quality: int = 85
) -> Tuple[bytes, str]:
    """
    Compress an image by resizing and converting to JPEG.

    Args:
        image_data: Raw image bytes
        max_width: Maximum width in pixels
        jpeg_quality: JPEG quality (1-100)

    Returns:
        Tuple of (compressed bytes, mime type)
    """
    with Image.open(BytesIO(image_data)) as img:
        # Convert RGBA to RGB (JPEG doesn't support alpha)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if wider than max_width
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Save as JPEG
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
        buffer.seek(0)

        return buffer.read(), 'jpeg'


def process_html_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    max_width: int = 1200,
    jpeg_quality: int = 85,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    Process a single HTML file, compressing embedded images.

    Args:
        input_path: Path to input HTML file
        output_path: Path to output file (if None, uses input_path with .compressed suffix)
        max_width: Maximum image width
        jpeg_quality: JPEG quality
        dry_run: If True, only calculate size reduction without saving

    Returns:
        Tuple of (original_size, new_size, images_processed)
    """
    logger.info(f"Processing: {input_path}")

    # Read original file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_size = len(content.encode('utf-8'))
    images_processed = 0

    def replace_image(match):
        nonlocal images_processed

        image_type = match.group(1).lower()
        base64_data = match.group(2)

        try:
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            original_img_size = len(image_bytes)

            # Compress
            compressed_bytes, new_type = compress_image(
                image_bytes, max_width, jpeg_quality
            )
            compressed_size = len(compressed_bytes)

            # Only replace if we achieved compression
            if compressed_size < original_img_size:
                images_processed += 1
                new_base64 = base64.b64encode(compressed_bytes).decode('ascii')
                reduction = (1 - compressed_size / original_img_size) * 100
                logger.debug(
                    f"  Image {images_processed}: {original_img_size/1024:.0f}KB -> "
                    f"{compressed_size/1024:.0f}KB ({reduction:.0f}% reduction)"
                )
                return f"data:image/{new_type};base64,{new_base64}"
            else:
                # Keep original if compression didn't help
                return match.group(0)

        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            return match.group(0)

    # Replace all images
    new_content = IMAGE_PATTERN.sub(replace_image, content)
    new_size = len(new_content.encode('utf-8'))

    # Save if not dry run
    if not dry_run and images_processed > 0:
        if output_path is None:
            # Rename original to .raw.html, save compressed as original name
            raw_path = input_path.with_suffix('.raw.html')
            input_path.rename(raw_path)
            logger.info(f"  Renamed original to: {raw_path}")
            output_path = input_path  # Use original filename for compressed

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        logger.info(f"  Saved to: {output_path}")

    return original_size, new_size, images_processed


def find_html_files(path: Path, recursive: bool = False) -> List[Path]:
    """Find HTML report files in a directory."""
    if path.is_file():
        if path.suffix.lower() == '.html' and not path.stem.endswith('.raw'):
            return [path]
        return []

    pattern = '**/pipeline_report.html' if recursive else '*/pipeline_report.html'
    files = list(path.glob(pattern))

    # Also check for report.html
    pattern2 = '**/report.html' if recursive else '*/report.html'
    files.extend(path.glob(pattern2))

    # Filter out already processed raw files
    files = [f for f in files if not f.stem.endswith('.raw')]

    return sorted(set(files))


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def main():
    parser = argparse.ArgumentParser(
        description='Compress HTML reports by converting embedded images to JPEG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Path to HTML file or directory containing HTML files'
    )
    parser.add_argument(
        '--max-width', '-w',
        type=int,
        default=1200,
        help='Maximum image width in pixels (default: 1200)'
    )
    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=85,
        help='JPEG quality 1-100 (default: 85)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search directories recursively'
    )
    parser.add_argument(
        '--in-place', '-i',
        action='store_true',
        help='Overwrite original files instead of creating .compressed.html'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show expected size reduction without modifying files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not PIL_AVAILABLE:
        logger.error("PIL/Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

    # Validate arguments
    if not args.path.exists():
        logger.error(f"Path does not exist: {args.path}")
        sys.exit(1)

    if args.quality < 1 or args.quality > 100:
        logger.error("Quality must be between 1 and 100")
        sys.exit(1)

    if args.max_width < 100:
        logger.error("Max width must be at least 100 pixels")
        sys.exit(1)

    # Find files to process
    files = find_html_files(args.path, args.recursive)

    if not files:
        logger.warning("No HTML files found to process")
        sys.exit(0)

    logger.info(f"Found {len(files)} HTML file(s) to process")
    logger.info(f"Settings: max_width={args.max_width}, quality={args.quality}")
    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")

    # Process files
    total_original = 0
    total_new = 0
    total_images = 0

    for html_file in files:
        output_path = html_file if args.in_place else None

        try:
            orig_size, new_size, img_count = process_html_file(
                html_file,
                output_path,
                args.max_width,
                args.quality,
                args.dry_run
            )

            total_original += orig_size
            total_new += new_size
            total_images += img_count

            reduction = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
            logger.info(
                f"  {format_size(orig_size)} -> {format_size(new_size)} "
                f"({reduction:.0f}% reduction, {img_count} images)"
            )

        except Exception as e:
            logger.error(f"Failed to process {html_file}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:     {len(files)}")
    print(f"Images compressed:   {total_images}")
    print(f"Original total size: {format_size(total_original)}")
    print(f"New total size:      {format_size(total_new)}")
    if total_original > 0:
        reduction = (1 - total_new / total_original) * 100
        saved = total_original - total_new
        print(f"Space saved:         {format_size(saved)} ({reduction:.0f}%)")

    if args.dry_run:
        print("\nDRY RUN - no files were modified")
        print("Run without --dry-run to apply compression")


if __name__ == '__main__':
    main()
