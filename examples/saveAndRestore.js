function count(res) {
    let count = 0;
    for (let k = 0; k < res.length; k++) {
        for (let q = 0; q < res[k].length; q++) {
            for (let i = 0; i < res[k][q].length; i++) {
                if (res[k][q][i] != null)
                    count += 1;
            }
        }
    }
    return count;
}

let start;
let engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("### WARMUP END ###")

start = performance.now();
let file = engine.load("../datasets/twitter_small_records.json", true);
let index = file.index(3, true);
let results = index.query("$.user.lang");
console.log("index: " + count(results) + " results");

index.save("index.test");
index.free();

let savedIndex = engine.restore("../datasets/twitter_small_records.json", "index.test");
let savedResults = savedIndex.query("$.user.lang");
console.log("saved index: " + count(savedResults) + " results");

savedIndex.free()