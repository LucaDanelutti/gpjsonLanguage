# The gpjson component for GraalVM

Truffle languages can be packaged as components which can be installed into
GraalVM using the [Graal
updater](http://www.graalvm.org/docs/reference-manual/graal-updater/). 
Running `mvn package` in the gpjsonLanguage folder also builds a
`gpjson-component.jar`. 
This file is the gpjson component for GraalVM and can be installed by
running:

```
/path/to/graalvm/bin/gu install /path/to/gpjson-component.jar
```

