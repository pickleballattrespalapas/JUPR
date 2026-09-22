const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const cache = new Map();
const files = {
  '@/components/SearchablePlayerSelect': 'components/SearchablePlayerSelect.tsx',
  '@/components/PlayerSearchInput': 'components/PlayerSearchInput.tsx',
  '@/components/PublicPlayerSearch': 'components/PublicPlayerSearch.tsx',
  '@/lib/playerSearch': 'lib/playerSearch.ts',
  '@/lib/api': 'lib/api.ts',
};
module.exports = function load(name) {
  if (!files[name]) return null;
  if (cache.has(name)) return cache.get(name);
  const module = { exports: {} };
  const source = fs.readFileSync(path.join(__dirname, '../..', files[name]), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function('require', 'module', 'exports', code)(dependency => load(dependency) || require(dependency), module, module.exports);
  cache.set(name, module.exports);
  return module.exports;
};
