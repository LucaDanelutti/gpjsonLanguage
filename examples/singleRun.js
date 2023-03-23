let start;
var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

start = performance.now();
let results = engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true, false);
console.log("Total time: " + (performance.now() - start) + "ms");

for (let i=0; i<5; i++) {
    console.log(results[0][i]);
}

function sleep(millis) {
    return new Promise(resolve => setTimeout(resolve, millis));
}
/*
sleep(20*1000).then(() => {
    console.log("Exiting")
});*/
