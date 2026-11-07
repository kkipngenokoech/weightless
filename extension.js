const vscode = require('vscode');
const path = require('path');
const { loadModel } = require('./lib/safetensors');
const { loadGGUFModel } = require('./lib/gguf');
const { loadFromHub } = require('./lib/hub');

function getNonce() {
  let text = '';
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) text += chars.charAt(Math.floor(Math.random() * chars.length));
  return text;
}

// anatomy → gallery data → explain views → the viewer shell (order matters)
const SCRIPTS = ['anatomy.js', 'gallery.js', 'explain.js', 'main.js'];

function getHtml(webview, extUri, nonce) {
  const uri = (f) => webview.asWebviewUri(vscode.Uri.joinPath(extUri, 'media', f));
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="${uri('style.css')}" rel="stylesheet">
  <link href="${uri('wl.css')}" rel="stylesheet">
  <title>Weightless</title>
</head>
<body class="wl-body">
  <div id="app"><div class="loading">Reading header…</div></div>
  ${SCRIPTS.map((f) => `<script nonce="${nonce}" src="${uri(f)}"></script>`).join('\n  ')}
</body>
</html>`;
}

function loadByExt(fsPath) {
  return /\.gguf$/i.test(fsPath) ? loadGGUFModel(fsPath) : loadModel(fsPath);
}

class SafetensorsEditorProvider {
  constructor(context) { this.context = context; }
  async openCustomDocument(uri) { return { uri, dispose() {} }; }
  async resolveCustomEditor(document, webviewPanel) {
    const webview = webviewPanel.webview;
    webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, 'media')],
    };
    webview.html = getHtml(webview, this.context.extensionUri, getNonce());
    try {
      const model = await loadByExt(document.uri.fsPath);
      webview.postMessage({ type: 'model', fileName: path.basename(document.uri.fsPath), ...model });
    } catch (err) {
      webview.postMessage({ type: 'error', message: String((err && err.message) || err) });
    }
  }
}

// Sidebar: lists every model file in the workspace; click to open in the viewer.
class ModelFilesProvider {
  constructor() {
    this._em = new vscode.EventEmitter();
    this.onDidChangeTreeData = this._em.event;
  }
  refresh() { this._em.fire(undefined); }
  getTreeItem(item) { return item; }
  async getChildren() {
    const files = await vscode.workspace.findFiles('**/*.{safetensors,gguf}', '**/node_modules/**', 500);
    return files
      .sort((a, b) => a.fsPath.localeCompare(b.fsPath))
      .map((uri) => {
        const item = new vscode.TreeItem(path.basename(uri.fsPath));
        item.description = vscode.workspace.asRelativePath(path.dirname(uri.fsPath));
        item.resourceUri = uri;
        item.iconPath = new vscode.ThemeIcon('file-binary');
        item.tooltip = uri.fsPath;
        item.command = { command: 'vscode.openWith', title: 'Open with Weightless', arguments: [uri, 'safetensorsViewer.preview'] };
        return item;
      });
  }
}

function activate(context) {
  const modelFiles = new ModelFilesProvider();
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider('weightless.models', modelFiles),
    vscode.commands.registerCommand('weightless.refreshModels', () => modelFiles.refresh())
  );
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      'safetensorsViewer.preview',
      new SafetensorsEditorProvider(context),
      { webviewOptions: { retainContextWhenHidden: true }, supportsMultipleEditorsPerDocument: false }
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand('safetensorsViewer.open', (uri) => {
      const target = uri || (vscode.window.activeTextEditor && vscode.window.activeTextEditor.document.uri);
      if (target) vscode.commands.executeCommand('vscode.openWith', target, 'safetensorsViewer.preview');
    })
  );
  // Hub panels: header-only Range requests, so even 100 GB models cost a few
  // hundred KB. The webview asks for models by id ({type:'open'}) from the gallery.
  function hubPanel(title) {
    const panel = vscode.window.createWebviewPanel(
      'safetensorsViewer.hub', title, vscode.ViewColumn.Active,
      { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')] }
    );
    panel.webview.html = getHtml(panel.webview, context.extensionUri, getNonce());
    panel.webview.onDidReceiveMessage((msg) => { if (msg && msg.type === 'open' && msg.id) openHub(panel, msg.id); });
    return panel;
  }
  async function openHub(panel, rawId) {
    const id = String(rawId).trim();
    panel.title = id.split('/').pop();
    panel.webview.postMessage({ type: 'loading', id });
    try {
      const model = await loadFromHub(id);
      panel.webview.postMessage({ type: 'model', fileName: id, ...model });
    } catch (err) {
      panel.webview.postMessage({ type: 'error', message: String((err && err.message) || err) });
    }
  }
  context.subscriptions.push(
    vscode.commands.registerCommand('weightless.browse', () => {
      const panel = hubPanel('Open models');
      panel.webview.postMessage({ type: 'gallery' });
    }),
    vscode.commands.registerCommand('safetensorsViewer.openFromHub', async () => {
      const id = await vscode.window.showInputBox({
        prompt: 'Hugging Face model id (weights are NOT downloaded — header only)',
        placeHolder: 'e.g. google/gemma-3-270m or Qwen/Qwen2.5-7B-Instruct',
        ignoreFocusOut: true,
      });
      if (id) openHub(hubPanel(`HF: ${id.trim()}`), id);
    })
  );
}

function deactivate() {}
module.exports = { activate, deactivate };
