# GpJSON - Leveraging Structural Indexes for High-Performance JSON Data Processing on GPUs
This Truffle language exposes JSONPath query execution to the polyglot [GraalVM](http://www.graalvm.org).

The goals are:
 1. Present a couple of parsing techniques based on structural indexes to quickly execute queries on JSON files
 2. Introduce a batching approach to improve performance and allow the processing of datasets bigger than the GPU’s memory
 3. Implement the above concepts into a Truffle Language to provide an engine that can be used from any host language that can run on the GraalVM.
 
## Using GpJSON in the GraalVM
To compile JAR file containing GpJSON move to the language folder and run ```mvn package```.

Next, copy the JAR file from ```target/gpjson.jar``` into `jre/languages/gpjson` (Java 8) or `languages/gpjson` (Java 11) of the Graal installation. 

Note that `--jvm` and `--polyglot` must be specified in both cases as well.

In the examples folder you can find a couple of files containing examples of the GpJSON's syntax.
