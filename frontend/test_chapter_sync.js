// Unit tests for getActiveChapter(currentTime, scenes)
// Verifies 1-based indexing, start times, exact boundary times, edge cases, and clamping.

const assert = require("assert");

function getActiveChapter(currentTime, scenes) {
  if (!scenes || scenes.length === 0) return 1;
  let chapter = 1;
  for (let i = 0; i < scenes.length; i++) {
    if (currentTime >= scenes[i].start) {
      chapter = i + 1;
    }
  }
  return Math.min(Math.max(chapter, 1), scenes.length);
}

const sampleScenes = [
  { start: 0.0, title: "1. Core Dilemma: Production vs Reproduction" },
  { start: 12.5, title: "2. The Three Contradictions of Capitalism" },
  { start: 28.0, title: "3. First Regime: 19th-Century Liberal Capitalism" },
  { start: 45.2, title: "4. Second Regime: Mid-20th-Century State-Managed" },
  { start: 62.0, title: "5. Third Regime: Neoliberal Financialization" },
  { start: 80.5, title: "6. Contemporary Double Movement" },
  { start: 98.0, title: "7. Strategic Takeaway: Realigning Economy and Care" },
];

console.log("Running getActiveChapter tests...");

// 1. Initial time and within first chapter
assert.strictEqual(getActiveChapter(0.0, sampleScenes), 1, "t=0.0 must be chapter 1");
assert.strictEqual(getActiveChapter(5.0, sampleScenes), 1, "t=5.0 must be chapter 1");
assert.strictEqual(getActiveChapter(12.499, sampleScenes), 1, "t=12.499 must still be chapter 1");

// 2. Exact boundary for chapter 2
assert.strictEqual(getActiveChapter(12.5, sampleScenes), 2, "t=12.5 exact boundary must be chapter 2");
assert.strictEqual(getActiveChapter(12.501, sampleScenes), 2, "t=12.501 must be chapter 2");
assert.strictEqual(getActiveChapter(20.0, sampleScenes), 2, "t=20.0 must be chapter 2");
assert.strictEqual(getActiveChapter(27.999, sampleScenes), 2, "t=27.999 must still be chapter 2");

// 3. Exact boundary for chapter 3
assert.strictEqual(getActiveChapter(28.0, sampleScenes), 3, "t=28.0 exact boundary must be chapter 3");
assert.strictEqual(getActiveChapter(35.0, sampleScenes), 3, "t=35.0 must be chapter 3");

// 4. Exact boundaries for chapter 4, 5, 6, 7
assert.strictEqual(getActiveChapter(45.2, sampleScenes), 4, "t=45.2 exact boundary must be chapter 4");
assert.strictEqual(getActiveChapter(62.0, sampleScenes), 5, "t=62.0 exact boundary must be chapter 5");
assert.strictEqual(getActiveChapter(80.5, sampleScenes), 6, "t=80.5 exact boundary must be chapter 6");
assert.strictEqual(getActiveChapter(98.0, sampleScenes), 7, "t=98.0 exact boundary must be chapter 7");

// 5. Past the end of video (clamped to max chapter)
assert.strictEqual(getActiveChapter(150.0, sampleScenes), 7, "t=150.0 must clamp to last chapter (7)");
assert.strictEqual(getActiveChapter(9999.0, sampleScenes), 7, "t=9999.0 must clamp to last chapter (7)");

// 6. Negative or pre-start timestamps (clamped to chapter 1)
assert.strictEqual(getActiveChapter(-1.0, sampleScenes), 1, "t=-1.0 must clamp to chapter 1");
assert.strictEqual(getActiveChapter(-100.0, sampleScenes), 1, "t=-100.0 must clamp to chapter 1");

// 7. Edge cases: empty array or null
assert.strictEqual(getActiveChapter(10.0, []), 1, "Empty scenes must safely return 1");
assert.strictEqual(getActiveChapter(10.0, null), 1, "Null scenes must safely return 1");

console.log("All getActiveChapter tests passed successfully! (14 assertions)");
