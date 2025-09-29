#!/usr/bin/env bash
# gzip_patches.sh
# 压缩一组 all_channel_patches.npy 文件为 .npy.gz，覆盖原文件

set -euo pipefail

files=(
  '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft_hb/D10_0/all_channel_patches.npy'
  '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft_hb/D10_0_0/all_channel_patches.npy'
  '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft_hb/D10_0_1/all_channel_patches.npy'
  '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft_hb/D10_0_1_test/all_channel_patches.npy'
  '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft_hb/D10_2/all_channel_patches.npy'
  '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft_hb/D10_3/all_channel_patches.npy'
  '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ft_hb/D11_0/all_channel_patches.npy'
)

for f in "${files[@]}"; do
  if [ -f "$f" ]; then
    echo "Compressing: $f"
    gzip -f "$f"
  else
    echo "Missing: $f" >&2
  fi
done