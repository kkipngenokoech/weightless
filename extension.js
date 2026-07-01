const vscode = require('vscode');
const path = require('path');
const { loadModel } = require('./lib/safetensors');

function getNonce() {
  let text = '';
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) text += chars.charAt(Math.floor(Math.random() * chars.length));
  return text;
}

function getHtml(webview, extUri, nonce) {
  const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(extUri, 'media', 'style.css'));
  const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(extUri, 'media', 'main.js'));
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="${styleUri}" rel="stylesheet">
  <title>Safetensors Viewer</title>
</head>
<body>
  <div id="app"><div class="loading">Reading header…</div></div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
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
      const model = await loadModel(document.uri.fsPath);
      webview.postMessage({ type: 'model', fileName: path.basename(document.uri.fsPath), ...model });
    } catch (err) {
      webview.postMessage({ type: 'error', message: String((err && err.message) || err) });
    }
  }
}

function activate(context) {
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
}

function deactivate() {}
module.exports = { activate, deactivate };
