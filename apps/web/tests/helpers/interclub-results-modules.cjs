const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const playerSearch = require('./player-search-modules.cjs');
const cache = new Map();
const css = new Proxy({}, { get: (_, key) => key === '__esModule' ? false : key });
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '../..', file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
  new Function('require', 'module', 'exports', code)(name => {
    if (name.endsWith('.module.css')) return css;
    if (name === '@/lib/interclubResultViews') return load('lib/interclubResultViews.ts');
    if (name === '@/lib/interclubPublic') return load('lib/interclubPublic.ts');
    if (name === '@/lib/interclubAwards') return load('lib/interclubAwards.ts');
    if (name === './PublicClubLink' || name === 'next/link') return { __esModule: true, default: ({children, ...props}) => require('react').createElement('a', props, children) };
    if (name === './PublicInterclubCompetition') return load('components/PublicInterclubCompetition.tsx');
    if (name === './SearchablePlayerSelect') return playerSearch('@/components/SearchablePlayerSelect');
    return require(name);
  }, module, module.exports);
  cache.set(file, module.exports); return module.exports;
}
module.exports = load;
