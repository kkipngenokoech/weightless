# Contributing to Weightless

Bug reports, model compatibility reports, documentation fixes and pull requests are welcome.

## Run locally

Use Node.js 20 or newer and VS Code.

```bash
git clone https://github.com/kkipngenokoech/weightless.git
cd weightless
npm test
```

Press F5 in VS Code to open an Extension Development Host. Open a `.safetensors` or `.gguf` file, or run **Weightless: Browse Open Models**.

The extension uses plain JavaScript with no build step. The browser viewer shares its UI code with the extension.

## Find the code

- `extension.js` registers the editor, sidebar and commands.
- `lib/` contains checkpoint parsers and Hugging Face loading.
- `media/` contains the shared interface and model analysis.
- `web/` contains the browser host.
- `test/` contains parser and interaction checks.

## Report a problem

[Open an issue](https://github.com/kkipngenokoech/weightless/issues/new) with the steps to reproduce, expected behavior, actual behavior, and your VS Code version. For model compatibility problems, include the public model ID or a small anonymized tensor header. Remove credentials and private model information before sharing files or logs.

## Submit a pull request

1. Fork the repository and create a branch for one focused change.
2. Make the change and update relevant checks for parsing or interaction behavior.
3. Run `npm test`. For UI changes, run the DOM checks below and check the interface in the Extension Development Host.
4. Open a pull request describing the problem, resulting behavior and checks performed. Include a screenshot for visual changes.

```bash
npm install --prefix /tmp/weightless-test --no-save --ignore-scripts jsdom@26.1.0
NODE_PATH=/tmp/weightless-test/node_modules node test/explain.js
NODE_PATH=/tmp/weightless-test/node_modules node test/workspace.js
```

Keep the extension and browser viewer working together. Use VS Code theme variables for extension colors and preserve header-only model loading.

Contributions are made under the repository's [MIT license](LICENSE).
