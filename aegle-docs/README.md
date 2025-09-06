# Aegle Documentation Website

This documentation website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## 🚀 Quick Start

### Installation

```bash
npm install
```

### Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## 📦 Deployment

This website is deployed to GitHub Pages at: **https://kimpenn.github.io/aegle/**

### Manual Deployment (Current Method)

To deploy the documentation website:

1. **Build the website locally** (optional, for testing):
   ```bash
   npm run build
   npm run serve  # Preview the built site locally
   ```

2. **Deploy to GitHub Pages**:
   ```bash
   npm run deploy
   ```

This command builds the website and pushes the static files to the `gh-pages` branch, which is automatically served by GitHub Pages.

### Prerequisites for Deployment

- Ensure you have push access to the repository
- Make sure your Git credentials are configured correctly
- The `docusaurus.config.js` is already configured with the correct GitHub Pages settings:
  - `url: 'https://kimpenn.github.io'`
  - `baseUrl: '/aegle/'`
  - `organizationName: 'kimpenn'`
  - `projectName: 'aegle'`

### Deployment Workflow

1. Make changes to documentation files
2. Test locally with `npm start`
3. Build and deploy with `npm run deploy`
4. The website will be updated at https://kimpenn.github.io/aegle/ within a few minutes

## 📁 Project Structure

```
docs/                 # Documentation files (Markdown)
static/              # Static assets (images, files)
  └── img/           # Images used in documentation
src/                 # React components and pages
docusaurus.config.js # Docusaurus configuration
sidebars.js         # Sidebar configuration
```
