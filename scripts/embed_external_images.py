#!/usr/bin/env python3
"""
Embed external image references as base64 in HTML reports.

This script processes HTML files to convert external image references
(e.g., <img src="plots/image.png">) into self-contained base64 data URIs.
This makes the HTML file fully portable for sharing with collaborators.

Usage:
    # Single file
    python scripts/embed_external_images.py /path/to/report.html

    # Directory (batch)
    python scripts/embed_external_images.py /path/to/out/analysis/ --recursive

    # Options
    --max-width 1200     # Max image width (default: 1200)
    --quality 85         # JPEG quality 1-100 (default: 85)
    --dry-run            # Show expected changes without modifying files

By default, the script renames the original to .raw.html and saves the
modified version as the original filename.
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

# Regex pattern to match external image src attributes (not base64)
# Matches: src="path/to/image.png" or src='path/to/image.jpg'
# Does NOT match: src="data:image/..." (already base64)
EXTERNAL_IMG_PATTERN = re.compile(
    r'<img\s+([^>]*?)src=["\'](?!data:)([^"\']+)["\']([^>]*?)>',
    re.IGNORECASE | re.DOTALL
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
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])
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


def load_and_embed_image(
    image_path: Path,
    max_width: int = 1200,
    jpeg_quality: int = 85
) -> Optional[str]:
    """
    Load an image file and return as base64 data URI.

    Args:
        image_path: Path to the image file
        max_width: Maximum width in pixels
        jpeg_quality: JPEG quality (1-100)

    Returns:
        Base64 data URI string or None if failed
    """
    if not image_path.is_file():
        logger.warning(f"Image file not found: {image_path}")
        return None

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        compressed_bytes, mime_type = compress_image(image_data, max_width, jpeg_quality)
        base64_data = base64.b64encode(compressed_bytes).decode('ascii')
        return f"data:image/{mime_type};base64,{base64_data}"

    except Exception as e:
        logger.warning(f"Failed to process image {image_path}: {e}")
        return None


def process_html_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    max_width: int = 1200,
    jpeg_quality: int = 85,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Process a single HTML file, embedding external images as base64.

    Args:
        input_path: Path to input HTML file
        output_path: Path to output file (if None, renames original to .raw.html)
        max_width: Maximum image width
        jpeg_quality: JPEG quality
        dry_run: If True, only count images without modifying

    Returns:
        Tuple of (images_found, images_embedded)
    """
    logger.info(f"Processing: {input_path}")

    # Read original file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    html_dir = input_path.parent
    images_found = 0
    images_embedded = 0

    def replace_external_image(match):
        nonlocal images_found, images_embedded

        prefix_attrs = match.group(1)
        src_path = match.group(2)
        suffix_attrs = match.group(3)

        images_found += 1

        # Skip if already a URL (http/https)
        if src_path.startswith(('http://', 'https://')):
            logger.debug(f"  Skipping URL: {src_path}")
            return match.group(0)

        # Resolve path relative to HTML file
        image_path = (html_dir / src_path).resolve()

        if dry_run:
            if image_path.is_file():
                logger.info(f"  Would embed: {src_path}")
                images_embedded += 1
            else:
                logger.warning(f"  Image not found: {src_path} -> {image_path}")
            return match.group(0)

        # Load and embed image
        base64_uri = load_and_embed_image(image_path, max_width, jpeg_quality)

        if base64_uri:
            images_embedded += 1
            original_size = image_path.stat().st_size if image_path.is_file() else 0
            new_size = len(base64_uri.split(',')[1]) * 3 // 4  # Approximate decoded size
            logger.debug(
                f"  Embedded: {src_path} ({original_size/1024:.0f}KB -> {new_size/1024:.0f}KB)"
            )
            return f'<img {prefix_attrs}src="{base64_uri}"{suffix_attrs}>'
        else:
            logger.warning(f"  Failed to embed: {src_path}")
            return match.group(0)

    # Replace all external images
    new_content = EXTERNAL_IMG_PATTERN.sub(replace_external_image, content)

    # Save if not dry run and we made changes
    if not dry_run and images_embedded > 0:
        if output_path is None:
            # Rename original to .raw.html, save modified as original name
            raw_path = input_path.with_suffix('.raw.html')
            if not raw_path.exists():  # Don't overwrite existing backup
                input_path.rename(raw_path)
                logger.info(f"  Renamed original to: {raw_path}")
            output_path = input_path

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        logger.info(f"  Saved to: {output_path}")

    return images_found, images_embedded


def find_html_files(path: Path, recursive: bool = False) -> List[Path]:
    """Find HTML report files in a directory."""
    if path.is_file():
        if path.suffix.lower() == '.html' and not path.stem.endswith('.raw'):
            return [path]
        return []

    # Analysis pipeline combined reports
    pattern = '**/pipeline_report_with_analysis.html' if recursive else '*/pipeline_report_with_analysis.html'
    files = list(path.glob(pattern))

    # Also check for analysis_highlights.html
    pattern2 = '**/analysis_highlights.html' if recursive else '*/analysis_highlights.html'
    files.extend(path.glob(pattern2))

    # Filter out already processed raw files
    files = [f for f in files if not f.stem.endswith('.raw')]

    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description='Embed external images as base64 in HTML reports',
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
        '--dry-run', '-n',
        action='store_true',
        help='Show expected changes without modifying files'
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
    total_found = 0
    total_embedded = 0

    for html_file in files:
        try:
            found, embedded = process_html_file(
                html_file,
                None,
                args.max_width,
                args.quality,
                args.dry_run
            )

            total_found += found
            total_embedded += embedded

            logger.info(f"  Found {found} external images, embedded {embedded}")

        except Exception as e:
            logger.error(f"Failed to process {html_file}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:      {len(files)}")
    print(f"External images found: {total_found}")
    print(f"Images embedded:       {total_embedded}")

    if args.dry_run:
        print("\nDRY RUN - no files were modified")
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
