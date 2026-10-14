const assert = require('node:assert/strict');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const Controls = require('./helpers/public-controls-module.cjs')('@/components/PublicControls').default;
const h = React.createElement;
const markup = renderToStaticMarkup(h('section', {},
  h(Controls, { context: 'Overall · Current season · Active players' }, h('form', {}, h('input', { name: 'q', defaultValue: 'Joe' }))),
  h('table', { 'aria-label': 'Standings' }, h('tbody', {}, h('tr', {}, h('td', {}, 'Joe'))))
));
assert.match(markup, /<summary[^>]*>View &amp; filters<\/summary>/);
assert.doesNotMatch(markup, /<details[^>]*\bopen/);
assert.ok(markup.indexOf('</details>') < markup.indexOf('data-public-view-context'), 'Selected context stays visible outside the disclosure');
assert.ok(markup.indexOf('</details>') < markup.indexOf('<table'), 'Primary standings remain visible when controls are collapsed');
assert.match(markup, /name="q" value="Joe"/, 'Collapsing preserves the submitted form values');
const setup = renderToStaticMarkup(h(Controls, { label: 'Set up matchup', initiallyOpen: true }, h('input', { name: 'player' })));
assert.match(setup, /<details[^>]*\bopen/);
console.log('Public controls: collapsed defaults, visible context/results, preserved form values, and required-setup exception passed.');
