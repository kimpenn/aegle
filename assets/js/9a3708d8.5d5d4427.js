"use strict";(self.webpackChunkaegle_docs=self.webpackChunkaegle_docs||[]).push([[543],{1665:(e,a,n)=>{n.r(a),n.d(a,{assets:()=>o,contentTitle:()=>l,default:()=>p,frontMatter:()=>r,metadata:()=>s,toc:()=>d});const s=JSON.parse('{"id":"post_analysis","title":"Post-Analysis Visualization","description":"Launch napari with the following data","source":"@site/docs/post_analysis.md","sourceDirName":".","slug":"/post_analysis","permalink":"/aegle/docs/post_analysis","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/post_analysis.md","tags":[],"version":"current","sidebarPosition":5,"frontMatter":{"sidebar_position":5},"sidebar":"tutorialSidebar","previous":{"title":"analysis","permalink":"/aegle/docs/Analysis/"}}');var i=n(4848),t=n(8453);const r={sidebar_position:5},l="Post-Analysis Visualization",o={},d=[];function c(e){const a={code:"code",h1:"h1",header:"header",p:"p",pre:"pre",...(0,t.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(a.header,{children:(0,i.jsx)(a.h1,{id:"post-analysis-visualization",children:"Post-Analysis Visualization"})}),"\n",(0,i.jsx)(a.p,{children:"Launch napari with the following data"}),"\n",(0,i.jsxs)(a.p,{children:[(0,i.jsx)(a.code,{children:"codex_analysis.h5ad"}),", ",(0,i.jsx)(a.code,{children:"original_seg_res_batch.pickle"}),", ",(0,i.jsx)(a.code,{children:"matched_seg_res_batch.pickle"}),", ",(0,i.jsx)(a.code,{children:"*.ome.tiff"})]}),"\n",(0,i.jsx)(a.p,{children:"Example ipynb code block"}),"\n",(0,i.jsx)(a.pre,{children:(0,i.jsx)(a.code,{className:"language-python",children:'import logging\nimport os\nimport sys\nimport anndata\nimport numpy as np\nimport pandas as pd\nfrom tifffile import imread\n\n# read anndata from h5ad\nfile_name = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/out/analysis/test_analysis/exp-1/codex_analysis.h5ad"\nadata = anndata.read_h5ad(file_name)\ncluster_int = adata.obs["leiden"].astype(int).values\n\n\npatch_index = 0\n# Read the original segmentation results\npkl_file = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/out/main/D18_Scan1_0/original_seg_res_batch.npy"\nlogging.info(f"[INFO] Loading codex_patches from {pkl_file}")\nseg_res_batch = np.load(pkl_file, allow_pickle=True)\nseg_data = seg_res_batch[patch_index]\ncell_mask = seg_data["cell"]\nnuc_mask = seg_data["nucleus"]\n\n# Read the repaired segmentation results\npkl_file = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/out/main/D18_Scan1_0/matched_seg_res_batch.npy"\nlogging.info(f"[INFO] Loading codex_patches from {pkl_file}")\nrepaired_seg_res_batch = np.load(pkl_file, allow_pickle=True)\nrepaired_seg_data = repaired_seg_res_batch[patch_index]\nrepaired_cell_mask = repaired_seg_data["cell_matched_mask"]\nrepaired_nuc_mask = repaired_seg_data["nucleus_matched_mask"]\n\n# Load the OME-TIFF image (you may need to provide full path)\nfile_name = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/D18_Scan1_tissue_0.ome.tiff"\nome_image = imread(file_name)  # shape might be (C, Z, Y, X) or (C, Y, X)\n# Adjust axis order if necessary (e.g., select 2D image or max projection)\n# Example: If image is (C, Y, X), and you want DAPI channel (say channel 0)\nimage_2d = ome_image[0]  # or np.max(ome_image, axis=0) for Z-projection\nimage_6 = ome_image[6]\nimage_7 = ome_image[7]\nimage_40 = ome_image[40]\n\n\nimport napari\nviewer = napari.Viewer()\nviewer.add_image(image_2d, name="DAPI / Tissue Image")\nviewer.add_image(image_6, name=\'Pan-Cytokeratin\')\nviewer.add_image(image_7, name=\'Collagen IV\')\nviewer.add_image(image_40, name=\'FUT4\')\n# Ensure the mask is integer type\ncell_mask_int = cell_mask.astype(np.int32)\nviewer.add_labels(cell_mask_int, name="Cell Mask")\n\n# Optional: Add nucleus mask if desired\nnuc_mask_int = nuc_mask.astype(np.int32)\nviewer.add_labels(nuc_mask_int, name="nucleus Mask")\n\n# Ensure the mask is integer type\nrepaired_cell_mask_int = repaired_cell_mask.astype(np.int32)\nviewer.add_labels(repaired_cell_mask_int, name="Cell Mask Reqaired")\n\n# Optional: Add nucleus mask if desired\nrepaired_nuc_mask_int = repaired_nuc_mask.astype(np.int32)\nviewer.add_labels(repaired_cell_mask_int, name="nucleus Mask Reqaired")\n\nnapari.run()\n'})})]})}function p(e={}){const{wrapper:a}={...(0,t.R)(),...e.components};return a?(0,i.jsx)(a,{...e,children:(0,i.jsx)(c,{...e})}):c(e)}},8453:(e,a,n)=>{n.d(a,{R:()=>r,x:()=>l});var s=n(6540);const i={},t=s.createContext(i);function r(e){const a=s.useContext(t);return s.useMemo((function(){return"function"==typeof e?e(a):{...a,...e}}),[a,e])}function l(e){let a;return a=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),s.createElement(t.Provider,{value:a},e.children)}}}]);