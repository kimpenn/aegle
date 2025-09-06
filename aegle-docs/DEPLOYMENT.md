# Documentation Auto-Deployment Setup Guide

This document explains how to set up automatic deployment so that remote services automatically rebuild and publish when documentation is updated.

## Option 1: GitHub Actions + GitHub Pages (Recommended)

### 1. Enable GitHub Pages

In your GitHub repository:
1. Go to `Settings` > `Pages`
2. In the "Source" section, select "GitHub Actions"
3. Save settings

### 2. Configure Workflow Permissions

In your GitHub repository:
1. Go to `Settings` > `Actions` > `General`
2. In the "Workflow permissions" section, select "Read and write permissions"
3. Check "Allow GitHub Actions to create and approve pull requests"
4. Save settings

### 3. Push Code

Push the code containing `.github/workflows/deploy.yml` to the main branch:

```bash
git add .
git commit -m "Add GitHub Actions workflow for documentation deployment"
git push origin main
```

### 4. Verify Deployment

- Check the "Actions" tab of your GitHub repository to confirm the workflow runs successfully
- Documentation will be automatically deployed to `https://kimpenn.github.io/aegle/`

## Option 2: Other Deployment Platforms

### Netlify
1. Connect your GitHub repository to Netlify
2. Set build command: `cd aegle-docs && npm run build`
3. Set publish directory: `aegle-docs/build`
4. Automatic deployment on every push to main branch

### Vercel
1. Connect your GitHub repository to Vercel
2. Set root directory to `aegle-docs`
3. Vercel will automatically detect the Docusaurus project and configure build settings

## Local Testing

Before pushing, it's recommended to test locally:

```bash
cd aegle-docs
npm install
npm run build
npm run serve
```

## Auto-Deployment Trigger Conditions

Currently configured trigger conditions:
- Push to main branch
- Only triggers when files in `aegle-docs/` folder change
- Pull Request to main branch will run build tests (but not deploy)

## Troubleshooting

### Build Failure
1. Check GitHub Actions logs
2. Confirm all image paths are correct
3. Confirm Markdown syntax is correct

### Deployment Failure
1. Check GitHub Pages settings
2. Confirm repository permissions are set correctly
3. Review Actions permission configuration

### Images Not Displaying
1. Confirm image files exist in `static/img/` directory
2. Check image paths use relative paths (e.g., `../../static/img/image.png`)
3. Confirm image files are committed to Git repository

## Manual Deployment (Backup Option)

If automatic deployment has issues, you can deploy manually:

```bash
cd aegle-docs
npm run build
npm run deploy
```

Note: Manual deployment requires configuring Git credentials and push permissions.
