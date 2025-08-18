'''
A script to analyze a multi-node ScoreP trace file, focusing on GPU metrics and kernel execution.
It reads the trace file, extracts GPU-related events, and attributes metrics to the corresponding GPU kernels
and threads on different nodes. The script generates a report and Gantt chart for the GPU metrics collected
during the execution of the application.

This is designed primarily for matching kernels to GPU energy consumption metrics
collected by ScoreP for ROCm-SMI compatible GPUs.

** Tested on Frontier with information collected with Mi250X GPU traces. **

Author: Adam McDaniel
Date: 2025-06-23
'''

from interval_timeline import MetricAttribution, Metric
import matplotlib.pyplot as plt
import otf2

# The trace file to analyze.
TRACE = "./scorep-traces/frontier-hpl-run-using-2-ranks-with-craypm/traces.otf2"

# This is a prefix for the node names in the cluster.
# This is used to identify the node name for each location.
# If the node name does not start with the given prefix denoted here,
# it might not be handled properly.
CLUSTER_NODE_NAME_PREFIX = 'frontier'

# A list of node names in the cluster for which we are collecting metrics.
# This will automatically be populated with the node names found in the trace.
NODES = []

# The list of threads that are assigned to each GPU on the node.
# In ScoreP, the first number in HIP[x:y] is the GPU number,
# and the second number is the stream number.
DEVICE_THREADS = {
    'GPU-0': ['HIP[0:0]', 'HIP[0:1]', 'HIP[0:2]', 'HIP[0:3]'],
    'GPU-1': ['HIP[1:0]', 'HIP[1:1]', 'HIP[1:2]', 'HIP[1:3]'],
    'GPU-2': ['HIP[2:0]', 'HIP[2:1]', 'HIP[2:2]', 'HIP[2:3]'],
    'GPU-3': ['HIP[3:0]', 'HIP[3:1]', 'HIP[3:2]', 'HIP[3:3]'],
    'GPU-0 Cray PM Counters (Energy)': [],
    'GPU-0 Cray PM Counters (Power)': [],
}

# A mapping from thread names (e.g., HIP[0:0]) to GPU names (e.g., GPU-0).
LOCATION_TO_DEVICE_NAME = {location: gpu for gpu, locations in DEVICE_THREADS.items() for location in locations}

# A map of metric names recorded by ScoreP to the GPUs they correspond to.
# Each of these metrics is collected for a specific device on each node
METRICS_TO_DEVICE = {
    # Each Mi250X GPU on Frontier has 2 GCDs, so we take the even numbered
    # metrics. The odd numbers always report zero.
    'A2rocm_smi:::energy_count:device=0': ('GPU-0', 'Joules', lambda x: x / 1_000_000),
    'A2rocm_smi:::energy_count:device=2': ('GPU-1', 'Joules', lambda x: x / 1_000_000),
    'A2rocm_smi:::energy_count:device=4': ('GPU-2', 'Joules', lambda x: x / 1_000_000),
    'A2rocm_smi:::energy_count:device=6': ('GPU-3', 'Joules', lambda x: x / 1_000_000),
    'A3coretemp:::craypm:accel0_energy': ('GPU-0 Cray PM Counters (Energy)', 'Joules', lambda x: x),
    'A3coretemp:::craypm:accel0_power': ('GPU-0 Cray PM Counters (Power)', 'Watts', lambda x: x),
}

# The initial GPU energy readings for each GPU on each node.
# This is used to calculate the energy consumed during the execution of the application.
# It is initialized to None for each GPU and will be set when the first metric event is encountered.
node_initial_gpu_energy = {}

# A dictionary to hold the metric attributions for each node.
# Each node will have its own MetricAttribution object that collects metrics for all GPUs on that node.
# The key is the node name, and the value is a MetricAttribution object.
# This allows us to collect metrics for multiple nodes in the same trace.
node_attributions = {}

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
        
        
        # Determine the node name from the location's group system tree parent.
        # For GPU events, it seems like the node name is stored an extra level up in the hierarchy.
        # (Maybe there's a better way to do this...)
        node_name = location.group.system_tree_parent.name
        if CLUSTER_NODE_NAME_PREFIX not in node_name:
            node_name = location.group.system_tree_parent.parent.name
            
        # If the node name is not already in the list of nodes, add it.
        # This allows us to dynamically discover nodes in the trace and attribute metrics for them.
        if node_name not in NODES:
            print("Identified new node:", node_name)
            NODES.append(node_name)
            node_attributions[node_name] = MetricAttribution(DEVICE_THREADS)
            node_initial_gpu_energy[node_name] = {device_info[0]: None for device_info in METRICS_TO_DEVICE.values()}
        
        if isinstance(event, otf2.events.Enter):
            # If the event is an Enter event, we record the entry into a function or region for a given device
            # `location.name` is the thread name (e.g., HIP[0:0]),
            # and `event.region.name` is the function or kernel name.
            node_attributions[node_name].enter(time=current_time, thread=location.name, function=event.region.name)
        elif isinstance(event, otf2.events.Leave):
            # If the event is a Leave event, we record the exit from a function or region for a given device
            # `location.name` is the thread name (e.g., HIP[0:0]), and we just record the time of the leave event
            # -- the callstack should already know where we are leaving from.
            node_attributions[node_name].leave(time=current_time, thread=location.name)
        elif isinstance(event, otf2.events.Metric) and event.member.name in METRICS_TO_DEVICE:
            # If we're recording a metric event that we want to attribute to a GPU kernel,
            # we first determine which GPU this metric corresponds to.
            # We first look at which GPU the metric is associated with using the `METRICS_TO_GPU` mapping.
            # The GPU name will be one of 'GPU-0', 'GPU-1', 'GPU-2, etc.
            device_name = METRICS_TO_DEVICE[event.member.name][0]
            unit = METRICS_TO_DEVICE[event.member.name][1]
            value_transform = METRICS_TO_DEVICE[event.member.name][2]
            
            # If the initial GPU energy for this node and GPU is not set, we set it to the current metric value.
            if node_initial_gpu_energy[node_name][device_name] is None:
                node_initial_gpu_energy[node_name][device_name] = value_transform(event.value)

            # We then sample the metric, subtracting the initial energy reading to get the energy consumed
            # since the last metric reading.
            # The value is divided by 1,000,000 to convert from microjoules to joules.
            node_attributions[node_name].sample(Metric(
                # The name of the metric, e.g., 'A2rocm_smi:::energy_count:device=0'
                name=event.member.name,
                # The value of the metric, adjusted by subtracting the initial energy reading for this GPU.
                value=value_transform(event.value) - node_initial_gpu_energy[node_name][device_name],
                # The time of the event in seconds since the start of the trace.
                # This is used to attribute the energy consumption for the active kernels and threads
                # when the metric was recorded.
                time=current_time,
                # The name of the device (e.g. 'frontier00000:HIP[0:0]') for which the metric is recorded.
                device=device_name,
                # The unit of the metric value, which is Joules in this case.
                unit="J",
            ))
print(f"Trace start time: {start_time / timer_resolution:.2f} seconds")
print(f"Trace end time: {end_time / timer_resolution:.2f} seconds")
print('Now reporting metrics for each node...')
# Report the attributions for each node and generate Gantt charts.
for node in NODES:
    attribution = node_attributions[node]
    print(f"Attribution for {node}:")
    print(attribution)
    print("\n")
    attribution.report()
    attribution.gantt_chart()
    
    
    # Now get the GPU samples for this node and graph them directly.
    metrics = [
        'A2rocm_smi:::energy_count:device=0',
        'A2rocm_smi:::energy_count:device=2',
        'A2rocm_smi:::energy_count:device=4',
        'A2rocm_smi:::energy_count:device=6',
        'A3coretemp:::craypm:accel0_energy',
        'A3coretemp:::craypm:accel0_power',
    ]

    samples = attribution.get_samples() # A dataframe with columns "Device", "Metric Name", "Value", "Time", "Unit"
    samples = samples[samples['Metric Name'].isin(metrics)]

    samples = samples.sort_values(['Device', 'Metric Name', 'Time'])
    # For each metric, get the value, time, and name as a list
    
    # Remove rows where the value is zero, as these are not useful for plotting.
    
    samples['Derived Power'] = samples.groupby(['Device', 'Metric Name'])['Value'].diff() / samples.groupby(['Device', 'Metric Name'])['Time'].diff()
    # Set the derived power to the value
    # Set all the derived power values for 'A3coretemp:::craypm:accel0_power' to 1
    samples.loc[samples['Metric Name'] == 'A3coretemp:::craypm:accel0_power', 'Derived Power'] = 1
    
    samples = samples[samples['Derived Power'] != 0]
    samples = samples.sort_values(['Device', 'Metric Name', 'Time'])
    samples['Derived Power'] = samples.groupby(['Device', 'Metric Name'])['Value'].diff() / samples.groupby(['Device', 'Metric Name'])['Time'].diff()
    samples.loc[samples['Metric Name'] == 'A3coretemp:::craypm:accel0_power', 'Derived Power'] = samples['Value']

    # Write the samples to a CSV file for further analysis if needed.
    samples.to_csv(f"{node}_gpu_samples.csv", index=False)
    # Write the samples to a CSV file for further analysis if needed.
    # samples.to_csv(f"{node}_gpu_samples.csv", index=False)
    
    # # Now graph the samples using a simple line plot. Show a different figure in the same window for each GPU.
    # fig, axes = plt.subplots(nrows=len(DEVICE_THREADS.keys()), figsize=(10, 5 * len(DEVICE_THREADS.keys())), sharex=True)
    # # Iterate over each GPU and plot its samples
    # for i, (gpu, group) in enumerate(samples.groupby('Device')):
    #     ax = axes[i]
    #     ax.plot(group['Time'], group['Value'], label=gpu, lw=1)
    #     ax.set_ylabel(f"{gpu} Energy (J)")
    #     ax.set_xlabel("Time (s)")
    #     ax.legend()
    #     ax.grid(True)

    # # Now graph the samples using a simple line plot. Show a different figure in the same window for each GPU.
    # fig, axes = plt.subplots(nrows=len(DEVICE_THREADS.keys()), figsize=(10, 5 * len(DEVICE_THREADS.keys())), sharex=True)
    # # Iterate over each GPU and plot its samples
    # for i, (gpu, group) in enumerate(samples.groupby('Device')):
    #     ax = axes[i]
    #     ax.plot(group['Time'], group['Derived Power'], label=gpu, lw=1)
    #     ax.set_ylabel(f"{gpu} Power (W)")
    #     ax.set_xlabel("Time (s)")
    #     ax.legend()
    #     ax.grid(True)
    
    # # Show the plot
    # plt.show()