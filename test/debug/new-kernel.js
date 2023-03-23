let start;
var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

let results = engine.query("new-kernel.json", ["$[?(@.name == 'Luca')].surname"], true, false);

for (let i=0; i<results[0].length; i++) {
    console.log(results[0][i]);
}

engine.close();