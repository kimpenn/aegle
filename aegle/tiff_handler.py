import re
from pathlib import Path
from typing import List

import numpy as np
import tifffile as tif


def fill_in_ome_meta_template(
    size_y: int, size_x: int, dtype, match_fraction: float
) -> str:
    template = """<?xml version="1.0" encoding="utf-8"?>
            <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
              <Image ID="Image:0" Name="mask.ome.tiff">
                <Pixels BigEndian="true" DimensionOrder="XYZCT" ID="Pixels:0" SizeC="4" SizeT="1" SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" Type="{dtype}">
                    <Channel ID="Channel:0:0" Name="cells" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:1" Name="nuclei" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:2" Name="cell_boundaries" SamplesPerPixel="1" />
                    <Channel ID="Channel:0:3" Name="nucleus_boundaries" SamplesPerPixel="1" />
                    <TiffData FirstC="0" FirstT="0" FirstZ="0" IFD="0" PlaneCount="1" />
                    <TiffData FirstC="1" FirstT="0" FirstZ="0" IFD="1" PlaneCount="1" />
                    <TiffData FirstC="2" FirstT="0" FirstZ="0" IFD="2" PlaneCount="1" />
                    <TiffData FirstC="3" FirstT="0" FirstZ="0" IFD="3" PlaneCount="1" />
                </Pixels>
              </Image>
              <StructuredAnnotations>
                <XMLAnnotation ID="Annotation:0">
                    <Value>
                        <OriginalMetadata>
                            <Key>FractionOfMatchedCellsAndNuclei</Key>
                            <Value>{match_fraction}</Value>
                        </OriginalMetadata>
                    </Value>
                </XMLAnnotation>
              </StructuredAnnotations>
            </OME>
        """
    ome_meta = template.format(
        size_y=size_y,
        size_x=size_x,
        dtype=np.dtype(dtype).name,
        match_fraction=match_fraction,
    )
    return ome_meta


def write_stack_to_file(out_path: str, stack, mismatch: float):
    dtype = np.uint32
    ome_meta = fill_in_ome_meta_template(
        stack.shape[-2], stack.shape[-1], dtype, mismatch
    )
    ome_meta_bytes = ome_meta.encode("UTF-8")
    stack_shape = stack.shape
    new_stack_shape = [stack_shape[0], 1, stack_shape[1], stack_shape[2]]
    with tif.TiffWriter(out_path, bigtiff=True, shaped=False) as TW:
        TW.write(
            stack.reshape(new_stack_shape).astype(dtype),
            contiguous=True,
            photometric="minisblack",
            description=ome_meta_bytes,
            metadata=None,
        )
