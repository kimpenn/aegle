---
sidebar_position: 2
---

# Aegle Alternative Setup (Docker via Enroot)

This guide describes how to run **Aegle** in an environment where **Visual Studio Code** or the **Dev Containers** extension cannot be used. While the guide describes a method that uses **Enroot** and **Pyxis** (for Slurm), users should adapt the below guide to their cluster environment to run the packaged Docker image from the GitHub Container Registry (GHCR).

---

## Prerequisites
- A remote Linux workstation or HPC cluster
- **Enroot** or equivalent container engine installed and configured on the host system
- (For Slurm clusters) **Pyxis** plugin enabled (optional for Enroot, but recommended for Slurm integration)
- **Git** installed and access to GitHub (usually via SSH)
- The raw image datasets
- Download zip file that contains `deepcell-pipeline` folder, `data` folder, and `bftools` folder from `https://kim.bio.upenn.edu/software/aegle/aegle_supplementary.zip` and unzip it to your project folder. You may discard `Dockerfile`, `devcontainer.json` files from the zip file as they are not used in the headless setup.

> Example working directory used below: `~/project/codex-analysis`

---
## Method A: Standalone Enroot (Workstation/Single Node)

### 1) Create a working directory
```bash
mkdir -p ~/project/codex-analysis
cd ~/project/codex-analysis
```
This folder will contain your raw data, pipeline repositories, supporting tools, and the imported container squash file.

---

### 2) Clone the Aegle repository
Create a subfolder in your working directory and clone the Aegle repository into it:
```bash
mkdir -p ~/project/codex-analysis/0-phenocycler-penntmc-pipeline
cd ~/project/codex-analysis/0-phenocycler-penntmc-pipeline
git clone git@github.com:kimpenn/aegle.git .
```

---

### 3) Add supporting pipeline folders
Copy the `deepcell-pipeline`, `bftools`, `data` folders into your working directory:
```bash
cp -r /path/to/deepcell-pipeline ~/project/codex-analysis/
cp -r /path/to/bftools ~/project/codex-analysis/
cp -r /path/to/data ~/project/codex-analysis/
```

Confirm that the Bio-Formats tools are executable:
```bash
chmod +x ~/project/codex-analysis/bftools/*
```

Also confirm that the following files have good permission (755):
```bash
`/workspaces/codex_analysis/bftools/showinf`
`/workspaces/codex_analysis/bftools/bf.sh`
```

---

### 4) Import the Docker image with Enroot
Use `enroot import` to download the prepackaged Docker image from GitHub Container Registry (GHCR):
```bash
enroot import -o ~/project/codex-analysis/aegle.sqsh docker://ghcr.io/seungyubh/aegle:latest
```

> **Tip:** If you do not specify the `-o` option, `enroot import` will download the image and create a squash file named `seungyubh+aegle+latest.sqsh` in your current working directory. You can rename it afterwards or specify it directly in the next step.

---

### 5) Create the Enroot container
Create an Enroot container named `aegle` from the squash file:
```bash
enroot create --name aegle ~/project/codex-analysis/aegle.sqsh
```

---


### 6) Create the data folder structure
Create the data directories needed by the pipeline stages:
```bash
mkdir -p ~/project/codex-analysis/data/deepcell
```

---

### 7) Start and interact with the container
Depending on your infrastructure, run the container either as a standalone Enroot session or via Slurm with Pyxis.

Start the Aegle container interactively, enable read-write access (`--rw`), and mount your working directory to the container path:
```bash
enroot start --rw --mount ~/project/codex-analysis/:/workspaces/codex-analysis aegle
```

> **Note:** Mounting to `/workspaces/codex-analysis` matches the standard workspace path used in the Dev Containers configuration. If you choose to mount to another path (e.g., `/workspaces`), adjust any paths in your pipeline configuration files accordingly.

---

### 8) Install Aegle in editable mode
Install the Aegle package within the container workspace by running:
```bash
cd /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline
python -m pip install -e .
```
---

## Method B: Slurm Cluster (via Pyxis)
If your HPC cluster runs Slurm and has the Pyxis plugin configured, you can launch jobs directly using the container image or your squash file without manually creating an Enroot container.

* **Interactive Session on a Compute Node:**
  ```bash
  srun --container-image=~/project/codex-analysis/aegle.sqsh \
       --container-mounts=~/project/codex-analysis:/workspaces/codex-analysis:workspaces:rw \
       --container-workdir=/workspaces \
       --container-name=aegle \
       --pty bash
  ```
  *(Alternatively, you can pull directly from GHCR: `--container-image=ghcr.io/seungyubh/aegle:latest`)*

---

## Verify the environment
Once inside the container (whether via Enroot or Pyxis), check that Python and other critical dependencies are accessible:
```bash
python --version
pip list | grep aegle
which git
```

If the Aegle package is not already installed in editable mode within the container workspace, run:
```bash
cd /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline
python -m pip install -e .
```

---

## Directory layout (after setup)
```
~/project/codex-analysis/
├── 0-phenocycler-penntmc-pipeline/
│   └── aegle/            # cloned repo contents
├── bftools/
├── deepcell-pipeline/
├── data/
│   └── deepcell/
└── aegle.sqsh            # imported squash file (optional to keep)
```

---

## Notes & Troubleshooting
- **GPU Access:** Enroot automatically integrates with NVIDIA GPUs if the NVIDIA container runtime/toolkit is installed on the host. When running under Slurm, Pyxis handles GPU isolation and mapping if you allocate GPUs via Slurm parameters (e.g. `--gpus=1`).
- **File Permissions:** Enroot containers execute in user namespaces. Unlike standard Docker containers which often run as root, any files generated by Aegle inside an Enroot session will be owned by your normal user account on the host, avoiding permission conflicts.
- **Docusaurus Links:** To find other configuration details, refer to the [Aegle Environment Setup](./aegle_environment_setup.md) guide.

---

## You're ready to run Aegle
Your headless environment is configured and the codebase is ready. You can now execute pipeline stages or queue cluster batch jobs using Slurm.
