const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
let loaded;
module.exports = function publicControls(name) {
  if (name !== '@/components/PublicControls') return null;
  if (!loaded) {
    const source = fs.readFileSync(path.join(__dirname, '../../components/PublicControls.tsx'), 'utf8');
    const code = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
    const module = { exports: {} };
    new Function('require', 'module', 'exports', code)(dependency => dependency.endsWith('.css') ? {} : require(dependency), module, module.exports);
    loaded = module.exports;
  }
  return loaded;
};
