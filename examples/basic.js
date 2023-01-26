var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

//warmup
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true);

let start = performance.now();
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true);
console.log("First query: " + (performance.now() - start) + "ms");

start = performance.now();
let results = engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true);
console.log("Second query: " + (performance.now() - start) + "ms");
for (let i=0; i<5; i++) {
	console.log(results[0][i][0]);
}
