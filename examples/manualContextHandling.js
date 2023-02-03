let engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

//warmup
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);

let start = performance.now();
let twitterContext = engine.createContext("../datasets/twitter_small_records.json");
twitterContext.loadFile();
twitterContext.buildIndexes(3, true);
twitterContext.query("$.user.lang");
twitterContext.free();
console.log("First query: " + (performance.now() - start) + "ms");

start = performance.now();
let twitterContext2 = engine.createContext("../datasets/twitter_small_records.json");
twitterContext2.loadFile();
twitterContext2.buildIndexes(3, true);
let results = twitterContext2.query("$.user.lang");
console.log("Second query: " + (performance.now() - start) + "ms");
for (let i=0; i<5; i++) {
    console.log(results[i][0]);
}

let twitterContext3 = engine.createContext("../datasets/twitter_small_records.json");
twitterContext3.free();
