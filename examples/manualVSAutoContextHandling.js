let engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

//warmup
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);

let start = performance.now();
for (let i=0; i<100; i++) {
    let twitterContext = engine.createContext("../datasets/twitter_small_records.json");
    twitterContext.loadFile();
    twitterContext.buildIndexes(3, true);
    twitterContext.query("$.user.lang");
}
let manual = (performance.now() - start);

start = performance.now();
for (let i=0; i<100; i++) {
    engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
}
let auto = (performance.now() - start);

console.log("Manual context queries: " + manual / 100 + "ms");
console.log("Auto context queries: " + auto / 100 + "ms");


