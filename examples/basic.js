var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

//warmup
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);

let start = performance.now();
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("First query: " + (performance.now() - start) + "ms");

start = performance.now();
let results = engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("Second query: " + (performance.now() - start) + "ms");
let count = 0;
for (let i=0; i<results[0].length; i++) {
    if (results[0][i] != null)
        count++;
}
console.log("Count: " + count);
