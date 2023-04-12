package it.necst.gpjson;

import org.graalvm.options.OptionCategory;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionStability;

import com.oracle.truffle.api.Option;

@Option.Group(GpJSONLanguage.ID)
public final class GpJSONOptions {

    @Option(category = OptionCategory.USER, help = "Set partition size for batch mode", stability = OptionStability.STABLE)
    public static final OptionKey<Integer> PartitionSize = new OptionKey<>((int) (1 * Math.pow(2, 30)));

    @Option(category = OptionCategory.USER, help = "Set stride for batch mode", stability = OptionStability.STABLE)
    public static final OptionKey<Integer> Stride = new OptionKey<>(8);

    @Option(category = OptionCategory.USER, help = "Set grid size for index construction", stability = OptionStability.STABLE)
    public static final OptionKey<Integer> IndexBuilderGridSize = new OptionKey<>(1024*16);

    @Option(category = OptionCategory.USER, help = "Set block size for index construction", stability = OptionStability.STABLE)
    public static final OptionKey<Integer> IndexBuilderBlockSize = new OptionKey<>(1024);

    @Option(category = OptionCategory.USER, help = "Set grid size for index construction", stability = OptionStability.STABLE)
    public static final OptionKey<Integer> QueryExecutorGridSize = new OptionKey<>(512);

    @Option(category = OptionCategory.USER, help = "Set block size for index construction", stability = OptionStability.STABLE)
    public static final OptionKey<Integer> QueryExecutorBlockSize = new OptionKey<>(1024);
}
