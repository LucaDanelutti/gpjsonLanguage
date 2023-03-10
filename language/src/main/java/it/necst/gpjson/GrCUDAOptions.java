package it.necst.gpjson;

import com.oracle.truffle.api.Option;
import org.graalvm.options.OptionCategory;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionStability;

@Option.Group("gpjson.grcuda")
public class GrCUDAOptions {
    @Option(category = OptionCategory.USER, help = "Log the execution time of GrCUDA computations using timers.", stability = OptionStability.STABLE) //
    public static final OptionKey<Boolean> EnableComputationTimers = new OptionKey<>(false);

    @Option(category = OptionCategory.USER, help = "Choose the scheduling policy of GrCUDA computations.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> ExecutionPolicy = new OptionKey<>("async");

    @Option(category = OptionCategory.USER, help = "Choose how data dependencies between GrCUDA computations are computed.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> DependencyPolicy = new OptionKey<>("with-const");

    @Option(category = OptionCategory.USER, help = "Choose how streams for new GrCUDA computations are created.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> RetrieveNewStreamPolicy = new OptionKey<>("always-new");

    @Option(category = OptionCategory.USER, help = "Choose how streams for new GrCUDA computations are obtained from parent computations.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> RetrieveParentStreamPolicy = new OptionKey<>("multigpu-disjoint");

    @Option(category = OptionCategory.USER, help = "Force the use of array stream attaching even when not required (e.g. post-Pascal GPUs).", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Boolean> ForceStreamAttach = new OptionKey<>(false);

    @Option(category = OptionCategory.USER, help = "Always prefetch input arrays to GPU if possible (e.g. post-Pascal GPUs).", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Boolean> InputPrefetch = new OptionKey<>(true);

    @Option(category = OptionCategory.USER, help = "Set how many GPUs can be used during computation. It must be at least 1, and if > 1 more than 1 GPUs are used (if available).", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<Integer> NumberOfGPUs = new OptionKey<>(1);

    @Option(category = OptionCategory.USER, help = "Choose the heuristic that manages how GPU computations are mapped to devices, if multiple GPUs are available.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> DeviceSelectionPolicy = new OptionKey<>("min-transfer-size");

    @Option(category = OptionCategory.USER, help = "Select a managed memory memAdvise flag, if multiple GPUs are available.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> MemAdvisePolicy = new OptionKey<>("none");

    @Option(category = OptionCategory.USER, help = "Add this option to dump scheduling DAG. Specify the destination path and the file name as value of the option (e.g. ../../../ExecutionDAG). File will be saved with .dot extension.", stability = OptionStability.EXPERIMENTAL) //
    public static final OptionKey<String> ExportDAG = new OptionKey<>("false");
}
