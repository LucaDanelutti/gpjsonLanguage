let start;
var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

start = performance.now();
engine.query("../datasets/twitter_small_records_2x.json", ["$.user.lang"], true, true);
console.log("Total time: " + (performance.now() - start) + "ms");

engine.close();

function sleep(millis) {
    return new Promise(resolve => setTimeout(resolve, millis));
}
/*
sleep(20*1000).then(() => {
    console.log("Exiting")
});*/