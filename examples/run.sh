js --polyglot --experimental-options \
  --log.gpjson.it.necst.gpjson.level=FINEST \
  --log.grcuda.com.nvidia.grcuda.level=INFO \
  --log.grcuda.com.nvidia.grcuda.runtime.executioncontext.level=OFF \
  --gpjson.grcuda.NumberOfGPUs=2 \
  --gpjson.grcuda.ExecutionPolicy=async \
  --gpjson.PartitionSize=$((2**30)) \
  $1
