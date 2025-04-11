# Aegle Pipeline

*Aegle* is a 2D data process and analysis pipeline for 2D PhenoCycler images. 

This pipeline is developed as part of the PennTMC project.  

![](aegle_icon.png)
## About the Name  

*Aegle* is named after the Greek goddess of radiant good health. Inspired by her name, this pipeline aims to bring clarity and precision to spatial biology by extracting meaningful insights from PhenoCycler images.  


## Key Features  

- **Image Preprocessing**  
  - **Tissue Extraction**: If a scan contains multiple tissues, `scripts/run_extract_tissue.sh` extracts tissues from the whole slide image (`.qptiff`) and saves them as separate images (`.ome.tiff`).  
  - **Antibody Extraction**: If antibody names are not provided with the scan, `scripts/run_extract_antibody.sh` retrieves antibody names from the metadata in `.qptiff` file and saves them as a CSV file.  

- **Cell Segmentation**  
  - Run segmentation with *Mesmer* using `scripts/run_main.sh` and do cell profiling to generate a cell x antibody matrix.

- **Downstream Analysis**  
  - Perform pixel-level quality control (QC), cell-level QC, clustering, maker test using `scripts/run_analysis.sh`.  

- **Patch Visualization**  
  - *Aegle* can segment images patch by patch. To visualize patch locations on the original image, use the standalong web app `aegle_patch_viewer`.  

- **Segmentation Evaluation**  
  - Assess and visualize segmentation quality using the [Napari Mask Density plugin](https://github.com/kuang-da/napari-mask-density), which provides intuitive overlays and density maps for inspecting segmentation performance.

## Development Status  

*Aegle* is under active development. The pipeline is not yet ready for production use, and we are continuously improving its features and performance.  

If you have any questions or feedback, please feel free to reach out to us.  

## Documentation
Comprehensive documentation for Aegle is currently under development and will be hosted at:
👉 https://kimpenn.github.io/aegle/

## Acknowledgements  

The *Aegle* pipeline builds upon and is inspired by the following projects. We greatly appreciate their contributions to the field. More detailed disclaimers and attributions will be included in the source code as development progresses.  

- [HuBMAP Hive phenocycler-pipeline](https://github.com/hubmapconsortium/phenocycler-pipeline)
- [HuBMAP Hive SPRM](https://github.com/hubmapconsortium/sprm)
- [CellSegmentationEvaluator](https://github.com/murphygroup/CellSegmentationEvaluator)
- [SPACEc](https://github.com/yuqiyuqitan/SPACEc)
- [DeepCell](https://github.com/vanvalenlab/deepcell-tf)
