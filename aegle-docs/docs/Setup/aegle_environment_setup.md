---
sidebar_position: 1
---

# Aegle Environment Setup

**Aegle** is packaged and distributed as a Development container. This guide walks you through the process of setting up the development container for a user to perform analysis on multiplexed imaging data from PhenoCycler/CODEX platforms.


---

## Prerequisites
- A Linux workstation or VM where Aegle will run (with sufficient CPU/RAM; GPU optional depending on your workflow)
- **Docker** installed and running
- (If using GPU) **NVIDIA drivers** and **NVIDIA Container Toolkit** installed
- **Visual Studio Code** with the **Dev Containers** extension on your local machine to connect to the workstation
- **Git** installed and access to GitHub via **SSH**
- Download `Dockerfile`, `devcontainer.json`, `deepcell-pipeline` folder, and `bftools` folder

> Example working directory used below: `~/project/codex-analysis`

---

## 1) Create a working directory
```bash
mkdir -p ~/project/codex-analysis
```

Within this directory you will keep your data, the dev container config, and pipeline repositories.

---

## 2) Add Dev Container configuration
Create a folder named **.devcontainer** inside the working directory and enter it:
```bash
mkdir -p ~/project/codex-analysis/.devcontainer
cd ~/project/codex-analysis/.devcontainer
```

Download and Copy the following two files into the `.devcontainer` folder:
- `Dockerfile`
- `devcontainer.json`

> **Tip:** If these live somewhere else on your machine or a repo, you can copy them here with `cp`.

### 2a) Modify configuration for your machine
Edit both files so paths and hardware options match your setup. Typical edits include:
- Base image and CUDA version (if using GPU)
- System packages and Python versions
- Mounts / volumes to expose the working directory
- User/group IDs if needed for file permissions

---

## 3) Add supporting pipeline folders
From your source location, copy the `deepcell-pipeline` folder into the working directory:
```bash
cp -r /path/to/deepcell-pipeline ~/project/codex-analysis/
```

---

## 4) Connect from your local machine with VS Code
1. Open VS Code on your local machine.
2. Remote into the target machine that will run Aegle (e.g., via SSH in VS Code or your preferred method).
3. In VS Code, open the working directory with **Command+O** (macOS) and select `~/project/codex-analysis`.

> **Tip:** If `~/project/codex-analysis` does not exist in the workstation, go to step 1 and follow the guide.

---

## 5) Build and enter the Dev Container
Use **Command+Shift+P** (macOS) and run: **Dev Containers: Rebuild and Reopen in Container**.

VS Code will build the Docker image as specified in `Dockerfile` and open the workspace inside the container.

---

## 6) Clone the Aegle repository
Inside the Dev Container (VS Code window says "Dev Container"), create a subfolder for the pipeline:
```bash
mkdir -p ~/project/codex-analysis/0-phenocycler-penntmc-pipeline
```

Then clone Aegle into that folder using VS Code:
1. **Command+Shift+P** → type **Git: Clone** → **Clone from GitHub**.
2. Paste the Aegle repo URL (SSH):
   ```
   git@github.com:kimpenn/aegle.git
   ```
3. Choose `~/project/codex-analysis/0-phenocycler-penntmc-pipeline` as the destination.

> You can also clone from a terminal inside the container:
> ```bash
> cd ~/project/codex-analysis/0-phenocycler-penntmc-pipeline
> git clone git@github.com:kimpenn/aegle.git .
> ```

---

## 7) Add Bio-Formats tools (bftools)
Copy the `bftools` folder into the **codex-analysis** dev container workspace:
```bash
cp -r /path/to/bftools ~/project/codex-analysis/
```

Confirm the tools are executable (adjust path as needed):
```bash
chmod +x ~/project/codex-analysis/bftools/*
```

---

## 8) Create the data folder structure
From a terminal **inside the Dev Container**:
```bash
mkdir -p ~/project/codex-analysis/data/deepcell
```

---

## 9) Verify the environment
- The VS Code window title should indicate you are inside the **Dev Container**.
- `docker ps` (outside the container) should show a running container for your workspace.
- Inside the container, confirm key tools are available:
```bash
python --version
pip list
which git
```

---

## Directory layout (after setup)
```
~/project/codex-analysis/
├── .devcontainer/
│   ├── Dockerfile
│   └── devcontainer.json
├── 0-phenocycler-penntmc-pipeline/
│   └── aegle/            # cloned repo contents
├── bftools/
├── deepcell-pipeline/
└── data/
    └── deepcell/
```

---

## Notes & Troubleshooting
- **Permissions:** If files created in the container appear owned by `root`, set `remoteUser` and/or `containerUser` in `devcontainer.json` and rebuild.
- **GPU access:** Ensure `--gpus all` (or the Dev Containers equivalent) is configured in `devcontainer.json` if required.
- **SSH keys:** If Git clone over SSH fails inside the container, share your SSH agent or copy your keys into the container securely.
- **Rebuilds:** Any changes to `Dockerfile` or `devcontainer.json` require **Dev Containers: Rebuild and Reopen in Container**.

---

## You’re ready to run Aegle
At this point, the Docker-based development environment is set up and the Aegle repository is cloned inside the container. Proceed with your usual Aegle workflow or pipeline commands within this environment.

