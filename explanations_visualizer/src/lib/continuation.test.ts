import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { appendPage, isComplete, shownText } from "./continuation.ts";

const noun = { one: "definition", many: "definitions" };
const key = (item: { id: string }) => item.id;
const page = (from: number, to: number) => Array.from({ length: to - from }, (_, index) => ({ id: `f${from + index}` }));

test("continuation pages append in order, keep earlier items and drop repeats (R06)", () => {
  const first = page(0, 20);
  const all = appendPage(first, page(18, 26), key);
  assert.equal(all.length, 26);
  assert.deepEqual(all.slice(0, 20), first);
  assert.deepEqual(all.map(key), page(0, 26).map(key));
  assert.equal(appendPage(all, page(0, 5), key), all, "a page of repeats changes nothing");
});

test("counts never claim completeness while more exist (R06)", () => {
  assert.equal(shownText(20, 26, true, noun), "20 of 26 definitions shown");
  assert.equal(shownText(20, null, true, noun), "20 definitions shown · more exist");
  assert.equal(shownText(20, 26, false, noun), "20 of 26 definitions shown", "a total above what is shown is not complete");
  assert.equal(shownText(26, 26, false, noun), "26 definitions");
  assert.equal(shownText(1, null, false, noun), "1 definition");
  assert.equal(isComplete(20, 26, true), false);
  assert.equal(isComplete(20, null, true), false);
  assert.equal(isComplete(20, 26, false), false);
  assert.equal(isComplete(26, 26, false), true);
  assert.equal(isComplete(3, null, false), true);
});
