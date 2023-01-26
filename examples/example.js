var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, 3);

let start = performance.now();
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, 3);
console.log(performance.now() - start);
start = performance.now();
let results = engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, 3);
console.log(performance.now() - start);
for (let i=0; i<20; i++) {
	console.log(results[0][i][0]);
}
