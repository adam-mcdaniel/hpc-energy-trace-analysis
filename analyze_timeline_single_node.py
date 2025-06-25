'''
A script to analyze a single-node trace from ScoreP, focusing on GPU metrics and kernel execution.
It reads the trace file, extracts GPU-related events, and attributes metrics to the corresponding GPU kernels
and threads. The script generates a report and Gantt chart for the GPU metrics collected during the
execution of the application.

This is designed primarily for matching kernels to GPU energy consumption metrics
collected by ScoreP for ROCm-SMI compatible GPUs.

** Tested with information collected with Mi300A GPU traces. **

Author: Adam McDaniel
Date: 2025-06-23
'''

from interval_timeline import MetricAttribution, Metric
import otf2

# The trace file to analyze.
TRACE = "./scorep-traces/simple-mi300-example-run/traces.otf2"

# The list of threads that are assigned to each GPU on the node.
# In ScoreP, the first number in HIP[x:y] is the GPU number,
# and the second number is the stream number. 
GPU_THREADS = {
    'GPU-0': ['HIP[0:0]', 'HIP[0:1]', 'HIP[0:2]', 'HIP[0:3]'],
    'GPU-1': ['HIP[1:0]', 'HIP[1:1]', 'HIP[1:2]', 'HIP[1:3]'],
    'GPU-2': ['HIP[2:0]', 'HIP[2:1]', 'HIP[2:2]', 'HIP[2:3]'],
    'GPU-3': ['HIP[3:0]', 'HIP[3:1]', 'HIP[3:2]', 'HIP[3:3]'],
}

# A mapping from thread names (e.g., HIP[0:0]) to GPU names (e.g., GPU-0).
LOCATION_TO_GPU_NAME = {location: gpu for gpu, locations in GPU_THREADS.items() for location in locations}

# A map of metric names recorded by ScoreP to the GPUs they correspond to.
# Each of these metrics is collected for a specific device on each node
METRICS_TO_GPU = {
    # There are 4 Mi300 GPUs, each in order
    f'A2rocm_smi:::energy_count:device=0': 'GPU-0',
    f'A2rocm_smi:::energy_count:device=1': 'GPU-1',
    f'A2rocm_smi:::energy_count:device=2': 'GPU-2',
    f'A2rocm_smi:::energy_count:device=3': 'GPU-3',
}

# The initial GPU energy readings for each GPU on.
# This is used to calculate the energy consumed during the execution of the application.
# It is initialized to None for each GPU and will be set when the first metric event is encountered.
initial_gpu_energy = {gpu: None for gpu in GPU_THREADS.keys()}

# The MetricAttribution object to collect metrics for all devices. This tracks calls and metrics,
# and attribute the metrics to the kernels / functions that are running on the devices.
attribution = MetricAttribution(GPU_THREADS)

# Initialize the start and end times for the trace.
# These will be used to calculate the time in seconds for each event.
start_time = None
end_time = None
with otf2.reader.open(TRACE) as reader:
    defs = reader.definitions
    # The timer resolution (ticks per second) of the trace.
    # This is used to convert the event times from ticks to seconds.
    timer_resolution = reader.timer_resolution

    for location, event in reader.events:
        # Check if the start time is not set, and if so, initialize it with the first event's time.
        # This is done to ensure that we have a reference point for the trace's time.
        if start_time is None:
            start_time = event.time  # Initialize start time on the first event
        # Update the end time to be the maximum of the current end time and the event's time.
        # This ensures that we capture the full duration of the trace.
        end_time = max(end_time, event.time) if end_time is not None else event.time
        
        # Get the current time in seconds by subtracting the start time from the event's time
        # and dividing by the timer resolution to convert from ticks to seconds.
        current_time = (event.time - start_time) / reader.timer_resolution  # Convert to seconds
        
        if isinstance(event, otf2.events.Enter):
            # If the event is an Enter event, we record the entry into a function or region for a given device
            # `location.name` is the thread name (e.g., HIP[0:0]),
            # and `event.region.name` is the function or kernel name.
            attribution.enter(time=current_time, thread=location.name, function=event.region.name)
        elif isinstance(event, otf2.events.Leave):
            # If the event is a Leave event, we record the exit from a function or region for a given device
            # `location.name` is the thread name (e.g., HIP[0:0]), and we just record the time of the leave event
            # -- the callstack should already know where we are leaving from.
            attribution.leave(time=current_time, thread=location.name)
        elif isinstance(event, otf2.events.Metric) and event.member.name in METRICS_TO_GPU:
            # If we're recording a metric event that we want to attribute to a GPU kernel,
            # we first determine which GPU this metric corresponds to.
            # We first look at which GPU the metric is associated with using the `METRICS_TO_GPU` mapping.
            # The GPU name will be one of 'GPU-0', 'GPU-1', 'GPU-2, etc.
            gpu_name = METRICS_TO_GPU.get(event.member.name, LOCATION_TO_GPU_NAME.get(location.name, "Unknown GPU"))
            
            # If the initial GPU energy for the GPU is not set, we set it to the current metric value.
            if initial_gpu_energy[gpu_name] is None:
                initial_gpu_energy[gpu_name] = event.value / 1000000
            
            # We then sample the metric, subtracting the initial energy reading to get the energy consumed
            # since the last metric reading.
            # The value is divided by 1,000,000 to convert from microjoules to joules.
            attribution.sample(Metric(
                # The name of the metric, e.g., 'A2rocm_smi:::energy_count:device=0'
                name=event.member.name,
                # The value of the metric, adjusted by subtracting the initial energy reading for this GPU.
                value=event.value / 1000000 - initial_gpu_energy[gpu_name],
                # The time of the event in seconds since the start of the trace.
                # This is used to attribute the energy consumption for the active kernels and threads
                # when the metric was recorded.
                time=current_time,
                # The name of the device (e.g. 'frontier00000:HIP[0:0]') for which the metric is recorded.
                device=gpu_name,
                # The unit of the metric value, which is Joules in this case.
                unit="J",
            ))

# Report the attributions for each device and generate Gantt charts.
attribution.report()
attribution.gantt_chart()