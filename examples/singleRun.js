var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

//warmup
engine.query("../datasets/twitter_small_records.json", ["$.user.lang"], true);
