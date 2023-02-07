var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

//warmup
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);

let start = performance.now();
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("First query: " + (performance.now() - start) + "ms");

start = performance.now();
let result = engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("Second query: " + (performance.now() - start) + "ms");
let count = 0;
for (let q = 0; q < result.length; q++) {
    for (let i = 0; i < result[q].length; i++) {
        for (let j = 0; j < result[q][i].length; j++) {
            if (result[q][i][j] != null)
                count += 1;
        }
    }
}
console.log("Count: " + count);

engine.close();
