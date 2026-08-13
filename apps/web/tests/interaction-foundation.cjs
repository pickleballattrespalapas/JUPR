const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const webRoot = path.resolve(__dirname, "..");
const read = (relativePath) => fs.readFileSync(path.join(webRoot, relativePath), "utf8");

const confirmAction = read("components/ConfirmAction.tsx");
const provider = read("components/interaction/InteractionProvider.tsx");
const dialog = read("components/interaction/InteractionDialog.tsx");
const lifecycle = read("components/interaction/useActionLifecycle.ts");
const types = read("components/interaction/types.ts");
const formDialog = read("components/interaction/FormDialog.tsx");
const feedback = read("components/interaction/ActionFeedback.tsx");
const css = read("components/interaction/InteractionDialog.module.css");

assert.match(types, /type ActionCompletion = ActionSuccess \| ActionUncertain/);
assert.match(types, /type ActionCallback = \(confirmationText: string\) => Promise<ActionCompletion>/);
assert.match(types, /function actionSuccess\(/);
assert.match(types, /function actionUncertain\(/);
assert.doesNotMatch(confirmAction, /Promise<void/);
assert.doesNotMatch(confirmAction, /window\.(confirm|alert|prompt)/);

assert.match(dialog, /<dialog/);
assert.match(dialog, /createPortal\(/);
assert.match(dialog, /document\.body/);
assert.match(dialog, /showModal\(\)/);
assert.match(dialog, /aria-busy=/);
assert.match(dialog, /onCancel=/);
assert.match(dialog, /event\.target === event\.currentTarget/);
assert.match(dialog, /returnFocusRef/);
assert.match(dialog, /originFocusRef/);
assert.match(dialog, /function isEligibleFocusTarget/);
assert.match(dialog, /!element\.matches\(":disabled"\)/);
assert.match(dialog, /element\.getAttribute\("aria-disabled"\) !== "true"/);
assert.match(dialog, /const returnTarget = returnFocusRef\?\.current \?\? null/);
assert.match(dialog, /isEligibleFocusTarget\(returnTarget\)/);
assert.match(dialog, /isEligibleFocusTarget\(rememberedFocusRef\.current\)/);
assert.match(dialog, /\.find\(\(element\) => isEligibleFocusTarget\(element\) && element\.tabIndex >= 0\)/);

assert.match(lifecycle, /useRef\(false\)/);
assert.match(lifecycle, /if \(inFlightRef\.current\) return null/);
assert.match(lifecycle, /setPhase\("working"\)/);
assert.match(lifecycle, /setPhase\(result\.status\)/);
assert.match(lifecycle, /normalizeInteractionActionError/);
assert.match(lifecycle, /if \(!isActionCompletion\(result\)\)/);
assert.match(lifecycle, /const recover = useCallback/);
assert.match(lifecycle, /setPhase\("uncertain"\)/);

assert.match(confirmAction, /useInteraction\(\)/);
assert.match(confirmAction, /openAction\(/);
assert.doesNotMatch(confirmAction, /useActionLifecycle/);
assert.match(provider, /Object\.freeze\(\{ \.\.\.request, origin \}\)/);
assert.match(provider, /activeRef\.current/);
assert.match(provider, /phase === "success"/);
assert.match(provider, /phase === "uncertain"/);
assert.match(provider, /lifecycle\.recover\(completion\.onRecover\)/);
assert.match(provider, /restoreFocus=\{false\}/);
assert.match(provider, /function isEligibleFocusTarget/);
assert.match(provider, /element\?\.isConnected/);
assert.match(provider, /!element\.matches\(":disabled"\)/);
assert.match(provider, /element\.getAttribute\("aria-disabled"\) !== "true"/);
assert.match(provider, /isEligibleFocusTarget\(explicitTarget\)/);
assert.match(provider, /isEligibleFocusTarget\(origin\)/);
assert.match(provider, /data-autofocus/);
assert.match(feedback, /role="alert"/);
assert.match(feedback, /data-dialog-focus/);
assert.match(feedback, /aria-label="Action result"/);
assert.doesNotMatch(feedback, /phase === "success"[\s\S]*?<h3[^>]*>\{completion\.title\}/);

assert.match(formDialog, /Discard unsaved changes\?/);
assert.match(formDialog, /getFirstInvalidField/);
assert.match(formDialog, /event\.preventDefault\(\)/);
assert.match(formDialog, /restoreFocus=\{lifecycle\.phase !== "success"\}/);
assert.match(formDialog, /originFocusRef=\{rememberedOriginRef\}/);
assert.match(formDialog, /focusEligibleElement\(explicitTarget\)/);
assert.match(formDialog, /focusEligibleElement\(rememberedOriginRef\.current\)/);
assert.doesNotMatch(formDialog, /finally \{\s*lifecycle\.reset\(\);\s*onCancel\(\)/, "success close must not re-enable competing dialog focus restoration");

assert.match(css, /min-height: 44px/);
assert.match(css, /@media \(max-width: 480px\)/);
assert.match(css, /calc\(100dvh - 1rem\)/);
assert.match(css, /outline: 3px solid #60a5fa/);

console.log("interaction foundation contract: ok");
