let start;
let engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("### WARMUP END ###")

start = performance.now();
let memory = engine.load("../datasets/twitter_small_records.json");
let index = engine.index(memory, 3, true);
let res = [];
for (let i=0; i<10; i++)
    res[i] = engine.query2(memory, index, "$.user.lang");
let time = (performance.now() - start) / 1000;


let count = 0;
for (let k = 0; k < 10; k++) {
    for (let q = 0; q < res[k].length; q++) {
        for (let i = 0; i < res[k][q].length; i++) {
            for (let j = 0; j < res[k][q][i].length; j++) {
                if (res[k][q][i][j] != null)
                    count += 1;
            }
        }
    }
}

console.log("1k queries: " + count + " results in " + time + "s");
engine.close();