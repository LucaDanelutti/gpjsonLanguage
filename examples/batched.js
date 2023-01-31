let start;
let numRuns = 1;
let engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

engine.query("../datasets/twitter_small_records_2x.json", ["$.user.lang"], true, false);

start = performance.now();
let batched;
for (let i=0; i<numRuns; i++)
    batched = engine.query("../datasets/twitter_small_records_2x.json", ["$.user.lang"], true, true);
let batchedTime = (performance.now() - start) / numRuns;
start = performance.now();
let block;
for (let i=0; i<numRuns; i++)
    block = engine.query("../datasets/twitter_small_records_2x.json", ["$.user.lang"], true, false);
let blockTime = (performance.now() - start) / numRuns;

let batchedCount = 0;
for (let i=0; i<batched[0].length; i++) {
    if (batched[0][i] != null)
        batchedCount++;
}
let blockCount = 0;
for (let i=0; i<block[0].length; i++) {
    if (block[0][i] != null)
        blockCount++;
}

console.log("Batched query: " + batchedCount + " results in " + batchedTime + "ms");
console.log("Block query: " + blockCount + " results in " + blockTime + "ms");