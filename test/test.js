const { assert } = require("console");
const fs = require("fs");
const jp = require("jsonpath");
const readline = require('readline');
var assertok = require('assert');

var engine = Polyglot.eval('gpjson', "GJ");
engine.buildKernels();

let DEBUG = true;

test("$.store.book[1].price");
// test("$.store.book[-1].price");
// test("$.store.book[-3].price");
test("$.store.book[0,2].category");
test("$.store.book[2,0].category");
test("$.store.book[2,1].category");
// test("$.store.book[2,2,2].category");
test("$.store.book[:3].category");
test("$.store.book[-3:].category");
test("$.store.book[3].category");
test("$.store.bicycle.color");
test("$.store.name");

async function test(query) {
    let gpjson = queryGPJSON(query);
    let node = await queryNode(query);

    if (DEBUG) {
        console.log(query + " -> " + gpjson.length + " -> " + gpjson + "|");
        console.log(query + " -> " + node.length + " -> " + node + "|");
    }

    assertok.ok(gpjson.length == node.length, "Fail: " + query);

    for (let i=0; i < gpjson.length; i++) {
        assertok.ok(gpjson[i].substring(1, gpjson[i].length-1) == node[i], "Fail: " + gpjson[i].substring(1, gpjson[i].length-1) + " != " + node[i]);
    }
}

function queryGPJSON(query) {
    let result = [];
    try {
        let value = engine.query("dataset.json", [query], true, false);
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

async function queryNode(query) {
    var lineReader = readline.createInterface({ input: fs.createReadStream("dataset.json") });
    var result = [];
    for await (const line of lineReader) {
        try {
            const obj = JSON.parse(line);
            let value = jp.query(obj, query);
            for (let item of value) {
                result.push(item);
            }
        } catch (error) {
            console.log(error);
        }
    }
    return result;
}