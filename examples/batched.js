let start;
let numRuns = 10;
let engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

engine.query("../datasets/twitter_small_records_2x.json", ["$.user.lang"], true, false);
console.log("### WARMUP END ###")

start = performance.now();
let block;
for (let i=0; i<numRuns; i++)
    block = engine.query("../datasets/twitter_small_records_2x.json", ["$.user.lang"], true, false);
let blockTime = (performance.now() - start) / numRuns;
console.log("### BLOCK END ###")
start = performance.now();
let batched;
for (let i=0; i<numRuns; i++)
    batched = engine.query("../datasets/twitter_small_records_2x.json", ["$.user.lang"], true, true);
let batchedTime = (performance.now() - start) / numRuns;
console.log("### BATCHED END ###")


let batchedCount = 0;
for (let q = 0; q < batched.length; q++) {
    for (let i = 0; i < batched[q].length; i++) {
        for (let j = 0; j < batched[q][i].length; j++) {
            if (batched[q][i][j] != null)
                batchedCount += 1;
        }
    }
}
let blockCount = 0;
for (let q = 0; q < block.length; q++) {
    for (let i = 0; i < block[q].length; i++) {
        for (let j = 0; j < block[q][i].length; j++) {
            if (block[q][i][j] != null)
                blockCount += 1;
        }
    }
}

console.log("Batched query: " + batchedCount + " results in avg " + batchedTime + "ms");
console.log("Block query: " + blockCount + " results in avg " + blockTime + "ms");