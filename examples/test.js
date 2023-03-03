var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

let res = engine.query("../datasets/test.json", ["$.surname"], true, false);

for (let q = 0; q < res.length; q++) {
    for (let i = 0; i < res[q].length; i++) {
        for (let j = 0; j < res[q][i].length; j++) {
            if (res[q][i][j] != null)
                process.stdout.write(res[q][i][j] + ", ");
        }
        process.stdout.write("\n");
    }
}