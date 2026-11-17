import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { savedChoiceFor } from "./sourceSelection.ts";

const choice = { package_id: "pkg", run_id: "run", source: "urn:source", source_kind: "object_property", query_id: 0, pair: "pair-zero" };

test("saved choices retain query zero and are isolated by package and run", () => {
  const raw = JSON.stringify(choice);
  assert.deepEqual(savedChoiceFor(raw, "pkg", "run"), choice);
  assert.equal(savedChoiceFor(raw, "other-package", "run"), null);
  assert.equal(savedChoiceFor(raw, "pkg", "other-run"), null);
  assert.equal(savedChoiceFor(raw, null, "run"), null);
  assert.equal(savedChoiceFor(raw, "pkg", null), null);
});

test("missing legacy query identity and source-only choices remain usable", () => {
  for (const query_id of [undefined, null]) {
    const saved = savedChoiceFor(JSON.stringify({ ...choice, source_kind: undefined, query_id, pair: "" }), "pkg", "run");
    assert.deepEqual(saved, { ...choice, source_kind: "class", query_id: null, pair: null });
  }
});

test("corrupt persisted queries cannot silently open another query for the same source", () => {
  for (const query_id of [-1, 0.5, "9", "", true, {}, 9007199254740992]) {
    assert.equal(savedChoiceFor(JSON.stringify({ ...choice, query_id }), "pkg", "run"), null);
  }
  for (const raw of [null, "", "{", "null", "[]", "false", JSON.stringify({ ...choice, source: "" })]) {
    assert.equal(savedChoiceFor(raw, "pkg", "run"), null);
  }
});
