import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { workspaceEvent } from "./workspaceEvents.ts";

test("opening the Hierarchy tab is not reported as a hierarchy expansion (C16)", () => {
  assert.notEqual(workspaceEvent({ type: "tab_open", tab: "hierarchy" })?.type, "hierarchy_expand");
  assert.equal(workspaceEvent({ type: "tab_open", tab: "evidence" })?.type, "table_open");
  assert.equal(workspaceEvent({ type: "tab_open", tab: "graph" })?.type, "graph_open");
});

test("expansion, navigation and inspection keep their own action types", () => {
  assert.deepEqual(workspaceEvent({ type: "hierarchy_expand", side: "source", iri: "https://example.org/a#X" }), { type: "hierarchy_expand", component: "hierarchy_source", element: "https://example.org/a#X" });
  assert.equal(workspaceEvent({ type: "hierarchy_focus", side: "target", iri: "x", kind: "class", via: "child" })?.type, "hierarchy_navigate");
  assert.equal(workspaceEvent({ type: "citation_open", factId: "f" })?.component, "citation");
  assert.equal(workspaceEvent({ type: "graph_reset" })?.type, "graph_reset");
});

test("a copied IRI or a search is never reported as external use", () => {
  assert.equal(workspaceEvent({ type: "copy_iri", side: "source" })?.type, "copy_iri");
  assert.equal(workspaceEvent({ type: "search", side: "source" })?.type, "search_select");
  assert.equal(workspaceEvent({ type: "search", side: "source" })?.element, undefined, "search text is never sent");
});
