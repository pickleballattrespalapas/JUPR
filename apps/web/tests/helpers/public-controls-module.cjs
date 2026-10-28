const fs = require('node:fs'), path = require('node:path'), ts = require('typescript');
const loaded = new Map();
module.exports = function publicControls(name) {
  const relative = {
    '@/components/GeneratorSessionDetails': '../../components/GeneratorSessionDetails.tsx',
    '@/components/PublicControls': '../../components/PublicControls.tsx',
    '@/lib/hubAutoAdvanceBoundary': '../../lib/hubAutoAdvanceBoundary.ts',
    '@/lib/generatorStandingsRequest': '../../lib/generatorStandingsRequest.ts',
  }[name];
  if (!relative) return null;
  if (!loaded.has(name)) {
    const source = fs.readFileSync(path.join(__dirname, relative), 'utf8');
    const code = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText;
    const module = { exports: {} };
    new Function('require', 'module', 'exports', code)(dependency => dependency.endsWith('.css') ? {} : require(dependency), module, module.exports);
    loaded.set(name, module.exports);
  }
  return loaded.get(name);
};
