let start;
let numRuns = 20;
var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

//warmup
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);

let result = new Array();
let count = new Array();
let time = new Array();

let startTotal = performance.now();
for (let k=0; k<numRuns; k++) {
    start = performance.now();
    result[k] = engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
    time[k] = (performance.now() - start);
}
console.log("Avg: " + (performance.now() - startTotal) / numRuns + "ms");

for (let k=0; k<numRuns; k++) {
    count[k] = 0;
    for (let q = 0; q < result.length; q++) {
        for (let i = 0; i < result[q].length; i++) {
            for (let j = 0; j < result[q][i].length; j++) {
                if (result[q][i][j] != null)
                    count[k] += 1;
            }
        }
    }
}

for (let i=0; i<numRuns; i++) {
    console.log(i + " query: " + count[i] + " in " + time[i] + "ms");
}
