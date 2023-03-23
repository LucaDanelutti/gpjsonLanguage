const { assert } = require("console");
var assertok = require('assert');

var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

let DEBUG = true;

test("$.store.book[1].price", ["8.99", "8.99"]);
test("$.store.book[-1].price", ["22.99", "22.99"]);
test("$.store.book[-3].price", ["8.95","8.95"]);
test("$.store.book[0,2].category", ["reference","fiction","reference","fiction"]);
test("$.store.book[2,0].category", ["fiction","reference","fiction","reference"]);
test("$.store.book[2,1].category", ["fiction","fiction","fiction","fiction"]);
test("$.store.book[2,2,2].category", ["fiction","fiction","fiction","fiction","fiction","fiction"]);
test("$.store.book[:3].category", ["reference","fiction","fiction","reference","fiction","fiction"]);
test("$.store.book[-3:].category", ["reference","fiction","fiction","reference","fiction","fiction"]);
test("$.store.book[3].category", []);
test("$.store.bicycle.color", ["red","red"]);
test("$.store.name", []);
test("$.store.bicycle[?(@.color == 'red')].price", ["19.95","19.95"]);

function test(query, results) {
    let gpjson = queryGPJSON(query);

    if (DEBUG) {
        console.log(query + " -> " + gpjson.length + " -> " + gpjson + "|");
        console.log(query + " -> " + results.length + " -> " + results + "|");
    }

    assertok.ok(gpjson.length == results.length, "Fail: " + query);

    for (let i=0; i < gpjson.length; i++) {
        assertok.ok(gpjson[i].substring(1, gpjson[i].length-1) == results[i], "Fail: " + gpjson[i].substring(1, gpjson[i].length-1) + " != " + results[i]);
    }
}

function queryGPJSON(query) {
    let result = [];
    try {
        let value = engine.query("test.json", [query], true, false);
        for (let i=0; i < value[0].length; i++) {
            for (let j = 0; j < value[0][i].length; j++) {
                if (value[0][i][j] != null)
                    result.push(value[0][i][j]);
            }
        }
    } catch (error) {
        console.log(error);
    }
    return result;
}