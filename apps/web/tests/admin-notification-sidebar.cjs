const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const ts = require('typescript'), React = require('react'), {create, act} = require('react-test-renderer');
function load(file, mocks = {}) {
  const code = ts.transpileModule(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), {compilerOptions: {target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true}}).outputText;
  const module = {exports: {}};
  new Function('require', 'module', 'exports', code)(name => Object.hasOwn(mocks, name) ? mocks[name] : require(name), module, module.exports);
  return module.exports;
}
const navigation = load('lib/adminNotificationNavigation.ts'), updates = load('lib/adminNotificationsEvents.ts');
const events = new Map(), documentEvents = new Map(), timers = new Map(); let timerId = 0;
const target = listeners => ({
  addEventListener(name, listener) {if (!listeners.has(name)) listeners.set(name, new Set()); listeners.get(name).add(listener);},
  removeEventListener(name, listener) {listeners.get(name)?.delete(listener);}
});
global.window = {...target(events), setInterval(callback, delay) {timers.set(++timerId, {callback, delay}); return timerId;}, clearInterval(id) {timers.delete(id);}};
global.document = {...target(documentEvents), visibilityState: 'visible'};
const item = (key, href, state = 'new', category = 'work') => ({key, href, state, category, kind: 'action', title: `Title ${key}`, description: 'Review pending work.', occurred_at: null});
const data = (club = 'alpha', items = []) => ({club_id: club, checked_at: '', history_days: 30, truncated: false, categories: [{key: 'work', enabled: true}, {key: 'off', enabled: false}], items});
const generator = item('generator', '/admin/play-generators/submissions?session=rr-1');
const registrations = item('tournament', '/admin/tournaments/fall?panel=registrations#approval');
const pool = item('pool', '/admin/interclub/seasons/season/clubs/alpha?panel=pool', 'flagged');
const groups = [
  {label: 'Workspace', links: [{href: '/admin', active: path => path === '/admin'}, {href: '/admin/notifications', active: () => false}, {href: '/admin/interclub', active: () => false}]},
  {label: 'Competitions', links: [{href: '/admin/play-generators', active: path => path.startsWith('/admin/play-generators')}, {href: '/admin/play-generators/submissions', active: () => false}, {href: '/admin/tournaments', active: path => path.startsWith('/admin/tournament-')}]}
];
function countsAreScopedToDestinationAndState() {
  const result = navigation.notificationNavigation(data('alpha', [generator, registrations, pool, item('cleared', '/admin/interclub', 'cleared'), item('off', '/admin/tournaments', 'new', 'off'), generator, item('alias', '/admin/tournament-checkin?event=fall'), item('prefix', '/admin/interclubbing')]), groups);
  assert.equal(result.byLink['/admin/notifications'].length, 5, 'Inbox contains unique active items in enabled categories');
  assert.equal(result.byLink['/admin/play-generators/submissions'].length, 1, 'Nested destination receives the badge');
  assert.equal(result.byLink['/admin/play-generators'], undefined, 'Parent destination is not counted twice');
  assert.equal(result.byLink['/admin/tournaments'].length, 2, 'Legacy tournament aliases retain the manager badge');
  assert.equal(result.byLink['/admin/interclub'].length, 1, 'Flagged items count, cleared items and lookalike prefixes do not');
  assert.equal(result.byGroup.Workspace.length, 1, 'Global inbox does not count unrelated Competition alerts in Workspace');
  assert.equal(result.byGroup.Competitions.length, 3);
  assert.match(navigation.notificationSummary([pool]), /1 active notification: Title pool/);
  assert.match(navigation.notificationSummary(Array.from({length: 6}, (_, i) => item(String(i), '/admin/interclub'))), /2 more/);
}
let request, requests = [];
const hook = load('lib/useAdminNotificationBadges.ts', {'./adminNotificationsApi': {getAdminNotifications: async (token, club) => {requests.push({token, club}); return request(token, club);}}, './adminNotificationsEvents': updates});
function Reader({token, club, enabled = true, pathname = '/admin'}) {
  const value = hook.useAdminNotificationBadges(token, club, pathname, enabled);
  return React.createElement('span', null, value ? value.items.map(item => item.title).join(',') : 'empty');
}
const content = tree => (tree.toJSON().children || []).join('');
const props = (token = 'token-a', club = 'alpha', extra = {}) => React.createElement(Reader, {token, club, ...extra});
const reply = value => ({data: value, status: 200, error: null});
async function scopedLiveUpdatesAndLateReads() {
  let resolveOld; request = () => new Promise(resolve => {resolveOld = resolve;}); let tree;
  await act(async () => {tree = create(props());}); assert.equal(content(tree), 'empty');
  await act(async () => updates.publishAdminNotifications('token-a', 'alpha', data('alpha', [pool]))); assert.equal(content(tree), 'Title pool');
  await act(async () => resolveOld(reply(data('alpha', [generator])))); assert.equal(content(tree), 'Title pool', 'Mutation update cannot be overwritten by an older sidebar read');
  await act(async () => updates.publishAdminNotifications('other-token', 'alpha', data('alpha', [generator])));
  await act(async () => updates.publishAdminNotifications('token-a', 'beta', data('beta', [generator])));
  assert.equal(content(tree), 'Title pool', 'Other users and clubs cannot supply sidebar data');
  await act(async () => updates.publishAdminNotifications('token-a', 'alpha', data('alpha', [])));
  assert.equal(content(tree), '', 'Clears update the mounted sidebar immediately');
  let resolveAlpha; request = () => new Promise(resolve => {resolveAlpha = resolve;});
  await act(async () => {for (const listener of events.get('focus')) void listener();});
  request = () => Promise.resolve(reply(data('beta', [registrations])));
  await act(async () => tree.update(props('token-b', 'beta'))); assert.equal(content(tree), 'Title tournament');
  await act(async () => resolveAlpha(reply(data('alpha', [pool]))));
  await act(async () => updates.publishAdminNotifications('token-a', 'alpha', data('alpha', [generator])));
  assert.equal(content(tree), 'Title tournament', 'Late reads and writes never expose previous club/user titles');
  await act(async () => tree.update(props('token-b', 'beta', {enabled: false}))); assert.equal(content(tree), 'empty');
  await act(async () => tree.unmount()); assert.equal(timers.size, 0);
  assert.ok([...events.values(), ...documentEvents.values()].every(listeners => listeners.size === 0));
}
async function focusRouteAndVisibilityRefresh() {
  requests = []; request = (token, club) => Promise.resolve(reply(data(club, [pool]))); let tree;
  await act(async () => {tree = create(props());}); assert.equal(requests.length, 1);
  await act(async () => tree.update(props('token-a', 'alpha', {pathname: '/admin/interclub'}))); assert.equal(requests.length, 2);
  document.visibilityState = 'hidden';
  await act(async () => {for (const listener of events.get('focus')) void listener();}); assert.equal(requests.length, 2);
  document.visibilityState = 'visible';
  await act(async () => {for (const listener of documentEvents.get('visibilitychange')) void listener();}); assert.equal(requests.length, 3);
  await act(async () => {for (const timer of timers.values()) void timer.callback();}); assert.equal(requests.length, 4);
  request = () => Promise.resolve({data: null, status: 403, error: 'Denied'});
  await act(async () => {for (const listener of events.get('focus')) void listener();}); assert.equal(content(tree), 'empty', 'Authorization denial removes stale titles');
  await act(async () => tree.unmount());
}
async function sidebarRendersLinkedAccessibleSummaries() {
  const workspace = {clubId: 'alpha', clubSlug: 'alpha'};
  const parents = [];
  document.body = {};
  window.innerWidth = 320;
  window.innerHeight = 300;
  const Shell = load('components/AdminShell.tsx', {
    'react-dom': {createPortal: children => React.createElement('div', {'data-tooltip-portal': true}, children)},
    'next/link': ({children, ...props}) => React.createElement('a', props, children),
    'next/navigation': {usePathname: () => '/admin/notifications', useRouter: () => ({replace() {}, refresh() {}})},
    '@/lib/adminAuthClient': {signOutAdminSession: async () => {}},
    '@/lib/useAdminSession': {useAdminSession: () => ({accessToken: 'token-a', loading: false, session: {user: {id: 'u1'}, capabilities: {assignments: [{club_id: 'alpha'}]}}})},
    '@/lib/useAvailableWorkspaces': {useAvailableWorkspaces: () => ({workspaces: [{club_id: 'alpha', club_slug: 'alpha', club_name: 'Alpha'}], loaded: true})},
    '@/lib/useAdminWorkspace': {AdminWorkspaceContext: React.createContext(workspace)},
    '@/lib/adminWorkspace': {canChooseAdminWorkspace: () => false, readBrowserWorkspace: () => workspace, sameWorkspace: () => true},
    '@/lib/adminNotificationNavigation': navigation,
    '@/lib/useAdminNotificationBadges': {useAdminNotificationBadges: () => data('alpha', [pool, registrations, generator])},
    './AdminShell.module.css': {}
  }).default;
  let tree; await act(async () => {tree = create(React.createElement(Shell, {workspace}, 'Content'), {createNodeMock(element) {
    if (element.props.role === 'tooltip') return {getBoundingClientRect: () => ({width: 280, height: 70})};
    if (element.type === 'span' && element.props.onMouseEnter) {
      const listeners = new Map(); parents.push(listeners);
      return {getBoundingClientRect: () => ({top: 285, bottom: 300, right: 300}), closest: () => ({
        addEventListener: (event, callback) => listeners.set(event, callback), removeEventListener: event => listeners.delete(event)
      })};
    }
    return null;
  }});});
  for (const [href, count, title] of [['/admin/notifications', 3, 'Title pool'], ['/admin/interclub', 1, 'Title pool'], ['/admin/tournaments', 1, 'Title tournament'], ['/admin/play-generators/submissions', 1, 'Title generator']]) {
    const link = tree.root.findAllByType('a').find(node => node.props.href === href);
    assert.ok(link.props['aria-describedby'], 'Native links expose their summary on keyboard focus');
    const summary = tree.root.findByProps({id: link.props['aria-describedby']});
    assert.match(summary.children.join(''), new RegExp(`${count} active notification`)); assert.match(summary.children.join(''), new RegExp(title));
  }
  const inbox = tree.root.findAllByType('a').find(node => node.props.href === '/admin/notifications');
  await act(async () => inbox.findAllByType('span').find(node => node.props.onMouseEnter).props.onMouseEnter());
  let popup = tree.root.findByProps({role: 'tooltip'});
  assert.deepEqual(popup.props.style, {top: 209, left: 20}, 'Summary opens above a badge at viewport bottom and stays within horizontal bounds');
  assert.equal(popup.parent.props['data-tooltip-portal'], true, 'Visual summary renders outside the scrollable sidebar');
  await act(async () => {for (const parent of parents) parent.get('keydown')?.({key: 'Escape'});});
  assert.equal(tree.root.findAllByProps({role: 'tooltip'}).length, 0, 'Escape dismisses the summary');
  await act(async () => parents[0].get('focus')());
  assert.equal(tree.root.findAllByProps({role: 'tooltip'}).length, 1, 'Keyboard focus displays a positioned summary');
  await act(async () => parents[0].get('blur')());
  assert.equal(tree.root.findAllByProps({role: 'tooltip'}).length, 0);
  const section = tree.root.findAllByType('button').find(node => node.props['aria-controls'] === 'admin-group-competitions');
  assert.match(tree.root.findByProps({id: section.props['aria-describedby']}).children.join(''), /^2 active notifications/);
  await act(async () => section.props.onClick()); assert.equal(section.props['aria-expanded'], false);
  assert.match(tree.root.findByProps({id: section.props['aria-describedby']}).children.join(''), /^2 active notifications/, 'Collapsed section keeps its count and summary');
  const unaffected = tree.root.findAllByType('a').find(node => node.props.href === '/admin/players'); assert.equal(unaffected.props['aria-describedby'], undefined);
  await act(async () => tree.unmount());
}
(async () => {
  countsAreScopedToDestinationAndState(); await scopedLiveUpdatesAndLateReads(); await focusRouteAndVisibilityRefresh(); await sidebarRendersLinkedAccessibleSummaries();
  console.log('Sidebar notifications: destination counts, flagged/cleared/preferences, scoped live updates, stale responses, refresh and accessible summaries passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
