let start;
let numRuns = 20;
var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

let dataset = "../datasets/twitter_small_records.json"
let queries = ["$.user.lang"]

//warmup
engine.query(dataset, queries, true, false);

let result = new Array();
let count = new Array();
let time = new Array();

let startTotal = performance.now();
for (let k=0; k<numRuns; k++) {
    start = performance.now();
    result[k] = engine.query(dataset, queries, true, false);
    time[k] = (performance.now() - start);
}
console.log("Avg: " + (performance.now() - startTotal) / numRuns + "ms");

for (let k=0; k<numRuns; k++) {
    count[k] = 0;
    for (let q = 0; q < result[k].length; q++) {
        for (let i = 0; i < result[k][q].length; i++) {
            for (let j = 0; j < result[k][q][i].length; j++) {
                if (result[k][q][i][j] != null)
                    count[k] += 1;
            }
        }
    }
}

for (let i=0; i<numRuns; i++) {
    console.log(i + " query: " + count[i] + " in " + time[i] + "ms");
}
