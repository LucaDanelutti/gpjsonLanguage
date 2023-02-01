let start;
var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

start = performance.now();
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("Total time: " + (performance.now() - start) + "ms");