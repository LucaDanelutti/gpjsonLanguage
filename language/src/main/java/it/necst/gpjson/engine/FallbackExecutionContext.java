package it.necst.gpjson.engine;

import com.jayway.jsonpath.*;
import it.necst.gpjson.GpJSONException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FallbackExecutionContext {
    private final String fileName;

    public FallbackExecutionContext(String fileName) {
        this.fileName = fileName;
    }

    public List<List<String>> execute(String query) {
        Configuration conf = Configuration.defaultConfiguration()
                .addOptions(Option.ALWAYS_RETURN_LIST, Option.ALWAYS_RETURN_LIST);
        ParseContext parseContext = JsonPath.using(conf);

        Path file = Paths.get(fileName);
        JsonPath compiledQuery = JsonPath.compile(query);

        try (Stream<String> lines = Files.lines(file, StandardCharsets.UTF_8)) {
            return lines.parallel().map(line -> {
                List<String> result;
                try {
                    result = Collections.singletonList(parseContext.parse(line).read(compiledQuery).toString());
                } catch (PathNotFoundException e) {
                    result = Collections.emptyList();
                }
                return result;
            }).collect(Collectors.toList());
        } catch (IOException e) {
            throw new GpJSONException("Failed to read file");
        }
    }
}
