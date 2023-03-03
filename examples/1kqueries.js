let start;
let numRuns = 1000;
let engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("### WARMUP END ###")

start = performance.now();
let file = engine.load("../datasets/twitter_small_records.json", true);
let index = file.index(3, true);
let res = [];
for (let i=0; i<numRuns; i++)
    res[i] = index.query("$.user.lang");
let time = (performance.now() - start) / 1000;


let count = 0;
for (let k = 0; k < numRuns; k++) {
    for (let q = 0; q < res[k].length; q++) {
        for (let i = 0; i < res[k][q].length; i++) {
            for (let j = 0; j < res[k][q][i].length; j++) {
                if (res[k][q][i][j] != null)
                    count += 1;
            }
        }
    }
}

index.free();
file.free();

console.log("1k queries: " + count + " results in " + time + "s");
engine.close();

function sleep(millis) {
    return new Promise(resolve => setTimeout(resolve, millis));
}

/*
sleep(20*1000).then(() => {
    console.log("Exiting")
});*/
