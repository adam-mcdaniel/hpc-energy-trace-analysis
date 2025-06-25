from typing import *
from collections import defaultdict
from dataclasses import dataclass
# gantt_helpers.py
from typing import List, Dict
import pandas as pd
import plotly.express as px
# from interval_timeline import MetricAttribution, Interval  # ya definidos en tu módulo
import unittest

class Interval(NamedTuple):
    start: float
    end: Optional[float] = None
    depth: float = 0
    name: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    def is_active(self, time: float) -> bool:
        '''
        Check if the interval is active at a given time.
        
        Parameters
        ----------
        time : int
            The time to check if the interval is active.
        
        Returns
        -------
        bool
            True if the interval is active at the given time, False otherwise.
        '''
        return self.start <= time and (self.end is None or time < self.end)
    
    def has_overlap(self, other: 'Interval') -> bool:
        '''
        Check if this interval overlaps with another interval.
        This does not return True if the intervals share an exact boundary,
        i.e., if the end of this interval is equal to the start of the other.
        It only returns True if there is an actual overlap.
        
        Parameters
        ----------
        other : Interval
            The other interval to check for overlap.
            
        Returns
        -------
        bool
            True if the intervals overlap, False otherwise.
        '''
        my_start = self.start
        my_end = self.end if self.end is not None else float('inf')
        other_start = other.start
        other_end = other.end if other.end is not None else float('inf')
        
        return my_start < other_end and other_start < my_end
    
    def clip(self, start: float, end: float) -> 'Interval':
        '''
        Clip the interval to the given start and end times.
        
        Parameters
        ----------
        start : float
            The new start time for the interval, if it lies after the current start.
        end : float
            The new end time for the interval, if it lies before the current end.
        
        Returns
        -------
        Interval
            A new interval with the clipped start and end times.
        '''
        if not self.has_overlap(Interval(start, end)):
            raise ValueError("The provided start and end times do not overlap with the interval.")
        
        new_start = max(self.start, start)
        new_end = min(self.end, end) if self.end is not None else end
        
        # Check if the interval is still valid after clipping
        # i.e., if the new end is not None and the new start is less than the new end.
        if new_end is not None and new_start > new_end:
            raise ValueError(f"Clipped interval has no valid duration (start >= end, {new_start} > {new_end}).")
        
        # Return a new Interval with the clipped times
        clipped_interval = self._replace(start=new_start, end=new_end)
        return clipped_interval

class Timeline:
    def __init__(self):
        self.finished_intervals: List[Interval] = []
        self.live_intervals: List[Interval] = []
        
    def enter(self, start: float, **kwargs):
        '''
        Start a new interval with the given parameters.
        
        Parameters
        ----------
        start : float
            The time at which the interval starts.
        
        **kwargs : dict
            Keyword arguments to initialize the interval.
            Optional keys include 'end', 'depth', 'name', and 'metadata'.
            
        Returns
        -------
        Interval
            The newly created interval.
        '''
        if 'end' in kwargs:
            raise ValueError("Cannot specify 'end' when entering a new interval.")
        interval = Interval(**kwargs, start=start, depth=len(self.live_intervals) + 1, metadata=kwargs.get('metadata', {}))
        self.live_intervals.append(interval)
        return interval
    
    def leave(self, end: float) -> None:
        '''
        Mark the most recent live interval as finished at the given time.
        
        Parameters
        ----------
        time : float
            The time at which the interval is finished.
            
        Raises
        ------
        ValueError
            If there are no active intervals to leave, or if the interval already has an end time.
            
        Returns
        -------
        None
        '''
        if not self.live_intervals:
            raise ValueError("No active intervals to leave.")
        
        interval = self.live_intervals.pop()
        if interval.end is not None:
            raise ValueError(f"Interval already has an end time of {interval.end}.")
        updated_interval = interval._replace(end=end)
        self.finished_intervals.append(updated_interval)
        
    def get_intervals_between(self, start: float, end: float) -> List[Interval]:
        '''
        Get all finished intervals that overlap with the given time range.
        Any interval which is live at the start time is also included, and
        its end time will be set to the end time of the range.
        All intervals that begin before the start time will also be clipped.
        
        Parameters
        ----------
        start : float
            The start time of the range.
        end : float
            The end time of the range.
        
        Returns
        -------
        List[Interval]
            A list of intervals that overlap with the given time range.
        
        Raises
        ------
        ValueError
            If the start time is greater than the end time.
        '''
        if start > end:
            raise ValueError("Start time must be less than or equal to end time.")
        
        # Filter the intervals that overlap with the specified start and end times
        overlapping_intervals = filter(
            lambda interval: interval.has_overlap(Interval(start, end)),
            self.finished_intervals + self.live_intervals
        )
        
        # Clip the intervals to the specified start and end times
        clipped_intervals = map(
            lambda interval: interval.clip(start, end),
            overlapping_intervals
        )
        
        # Sort them by start time, then by end time
        sorted_intervals = sorted(clipped_intervals, key=lambda x: (x.start, x.end if x.end is not None else float('inf')))
        
        return list(sorted_intervals)

class CallGraph(Timeline):
    def __init__(self):
        super().__init__()
    
    def exclusive_runtime(self, start: float, end: float=float('inf')) -> Dict[str, float]:
        """
        Calculate the exclusive runtime of each function over the time window [start, end].
        Exclusive runtime means the time a function spends running at the top of the call stack,
        excluding any time consumed by its child calls.

        Parameters
        ----------
        start : float
            Start of the time window.
        end : float
            End of the time window.

        Returns
        -------
        Dict[str, float]
            A mapping from function names to their exclusive runtimes.
        """
        if start > end:
            raise ValueError("Start time must be less than or equal to end time.")
        
        # 1. Retrieve and clip all intervals to the [start, end] window.
        intervals = self.get_intervals_between(start, end)
        for interval in intervals:
            assert interval.end is not None, "All intervals should have an end time after clipping."
        
        # 2. Build a list of timestamped “events”: (time, is_start, interval)
        events: List[Tuple[float, bool, Interval]] = []
        for iv in intervals:
            events.append((iv.start, True, iv))   # interval enters
            events.append((iv.end or float('inf'),   False, iv))  # interval leaves
        
        # 3. Sort events by time; for ties, start-events come before end-events
        events.sort(key=lambda e: (e[0], not e[1]))
        
        # 4. Sweep-line to accumulate exclusive time
        exclusive: Dict[str, float] = defaultdict(float)
        active: List[Interval] = []
        last_time: Optional[float] = None
        
        for timestamp, is_start, iv in events:
            if last_time is not None and timestamp > last_time:
                delta = timestamp - last_time
                if active:
                    # The deepest (max depth) active interval is the one running exclusively
                    deepest = max(active, key=lambda x: x.depth)
                    if deepest.name is not None:
                        exclusive[deepest.name] += delta
            
            last_time = timestamp
            if is_start:
                active.append(iv)
            else:
                active.remove(iv)
        
        return dict(exclusive)
    
def fraction_exclusive_runtime(exclusive_runtimes: List[Dict[str, float]]) -> Tuple[Dict[str, float], float]:
    fractions_of_exclusive_runtime = {}
    total_exclusive_runtime = 0.0
    for exclusive_runtime in exclusive_runtimes:
        total_exclusive_runtime += sum(exclusive_runtime.values())
        for function, runtime in exclusive_runtime.items():
            if function not in fractions_of_exclusive_runtime:
                fractions_of_exclusive_runtime[function] = 0.0
            fractions_of_exclusive_runtime[function] += runtime
    # Normalize the fractions
    if total_exclusive_runtime > 0:
        for function in fractions_of_exclusive_runtime:
            fractions_of_exclusive_runtime[function] /= total_exclusive_runtime
    return fractions_of_exclusive_runtime, total_exclusive_runtime

class Metric(NamedTuple):
    name: str
    value: float
    time: float
    unit: Optional[str] = None
    device: Optional[str] = None
    thread: Optional[str] = None
    
    def attribute(self, fraction: float, last_metric: 'Metric') -> 'Metric':
        """
        Attribute this metric by a fraction.
        
        Parameters
        ----------
        fraction : float
            The fraction to attribute the metric by.
        last_metric : Metric
            The last metric to use for context, if needed.
        
        Returns
        -------
        Metric
            A new Metric with the value adjusted by the fraction.
        """
        return self._replace(value=(self.value - last_metric.value) * fraction)
    
    def add(self, other: 'Metric') -> 'Metric':
        """
        Add two metrics together.
        
        Parameters
        ----------
        other : Metric
            The other metric to add.
        
        Returns
        -------
        Metric
            A new Metric with the combined value.
        """
        if self.name != other.name or self.unit != other.unit or self.device != other.device:
            raise ValueError("Metrics must have the same name and unit to be added.")
        return self._replace(value=self.value + other.value)
    
    def __str__(self):
        return f"{self.value:.6f} {self.unit}"
    
class MetricAttribution:
    def __init__(self, device_threads: Dict[str, List[str]]):
        self.device_threads = device_threads
        self.device_samples: Dict[str, List[Metric]] = {}
        self.thread_callgraphs: Dict[str, CallGraph] = defaultdict(CallGraph)
        self.function_attributions: Dict[str, Dict[str, Metric]] = defaultdict(dict)
        
    def initialize_metrics(self, metric: Metric, devices: List[str]):
        for device in devices:
            metric = metric._replace(device=device)
            self.device_samples.setdefault(device, []).append(metric)
    
    def enter(self, time: float, thread: str, function: str):
        self.thread_callgraphs[thread].enter(start=time, name=function)
        
    def leave(self, time: float, thread: str):
        if thread not in self.thread_callgraphs:
            raise ValueError(f"No call graph for thread {thread}.")
        self.thread_callgraphs[thread].leave(end=time)
        
    def sample(self, metric: Metric):
        """
        Sample a metric at a specific time.
        
        Parameters
        ----------
        metric : Metric
            The metric to sample.
        
        Returns
        -------
        None
        """
        # Get the device and thread from the metric
        device = metric.device if metric.device is not None else 'default'
        thread = metric.thread if metric.thread is not None else 'default'
        # Add device thread to device_threads if not seen before
        # if device not in self.device_threads:
        #     self.device_threads[device] = []
        # if thread not in self.device_threads[device]:
        #     self.device_threads[device].append(thread)
        
        if device not in self.device_samples:
            self.device_samples[device] = [metric]
        else:
            # First, get the last metric timestamp
            last_metric = self.device_samples[device][-1]
            last_metric_time = last_metric.time
            # Store the metric sample
            self.device_samples.setdefault(device, []).append(metric)
            
            # Get the time of the metric
            new_metric_time = metric.time
            
            fractions_of_runtime, total_exclusive_runtime = fraction_exclusive_runtime(
                [self.thread_callgraphs[thread].exclusive_runtime(start=last_metric_time, end=new_metric_time)
                for thread in self.device_threads.get(device, [])]
            )
            
            # Write the energy attribution back to the metadata
            for thread in self.device_threads.get(device, []):
                for iv in self.thread_callgraphs[thread].get_intervals_between(last_metric_time, new_metric_time):
                    # Interval runtime:
                    interval_end = iv.end if iv.end is not None else new_metric_time
                    interval_runtime = interval_end - iv.start
                    
                    # Add the metadata to the interval
                    if metric.name not in iv.metadata:
                        iv.metadata[metric.name] = metric.attribute(interval_runtime / total_exclusive_runtime, last_metric)
                    else:
                        iv.metadata[metric.name] = iv.metadata[metric.name].add(metric.attribute(interval_runtime / total_exclusive_runtime, last_metric))
                        
            # Now attribute the metric to each function based on the fraction of exclusive runtime
            for function, fraction in fractions_of_runtime.items():
                attributed_metric = metric.attribute(fraction, last_metric)
                if function not in self.function_attributions[device]:
                    self.function_attributions[device][function] = attributed_metric
                else:
                    self.function_attributions[device][function] = self.function_attributions[device][function].add(attributed_metric)

    def gantt_chart(self, clip_start: float = 0.0, clip_end: float = float("inf")):
        """
        Flatten all finished (and clipped-live) intervals from every thread/device
        into a tidy DataFrame suitable for px.timeline().
        """
        rows: List[Dict] = []

        for device, threads in self.device_threads.items():
            for thread in threads:
                callgraph = self.thread_callgraphs.get(thread)
                if callgraph is None:
                    continue  # no calls recorded for this thread

                # Clip to the requested window so the chart is not gigantic
                for iv in callgraph.get_intervals_between(clip_start, clip_end):
                    # Skip unfinished intervals (no end) because Gantt needs a Finish column
                    if iv.end is None:
                        continue
                    rows.append(
                        {
                            "Device": device,
                            "Thread": thread,
                            "Function": iv.name[:40] if iv.name else "Unnamed",
                            "Function Full": iv.name if iv.name else "Unnamed",
                            "Start": iv.start,
                            "Finish": iv.end,
                            "Depth": iv.depth,
                            "Metadata": ",".join([str(metadata_item) for metadata_item in iv.metadata.values()]) if iv.metadata else "",
                        }
                    )

        df = pd.DataFrame(rows)
        # Plotly will plot rows in the order given by y; a nicer default is alphabetical
        df.sort_values(["Device", "Thread", "Start"], inplace=True)
        
                
        df["StartDT"] = pd.to_datetime(df["Start"], unit="s")
        df["FinishDT"] = pd.to_datetime(df["Finish"], unit="s")

        # One Gantt per device in separate facets (handy if you have many threads/GPU queues)
        fig = px.timeline(
            df,
            x_start="StartDT",
            x_end="FinishDT",
            y="Thread",
            color="Function Full",
            facet_row="Device",
            hover_data={
                "Start": True,
                "Finish": True,
                "Function": True,
                "Function Full": False,
                "StartDT": False,  # ya lo tienes en %{x}; lo oculto aquí
                "FinishDT": False,  # ya lo tienes en %{x}; lo oculto
                "Metadata": True,
                "Thread": False,    # ya lo tienes en %{y}; lo oculto aquí
                "Function": True,  # no queremos ver ese nombre larguísimo
            },
        )

        # Improve aesthetics
        fig.update_yaxes(title_text="")
        fig.update_xaxes(
            # type="linear",          # ‹— clave: tratar X como valores lineales
            title="Time (s)",
        )

        fig.update_layout(
            showlegend=False,            # sin leyenda
            title="Metric Attribution",
            xaxis_title="Time (s)",
            legend_title="Function",
            bargap=0.0,
            height=400 + 200 * len(df["Device"].unique()),
        )

        # fig.update_layout(bargap=0.0)    # 5 % de hueco entre barras
        fig.show()
    
    def report(self):        
        exclusive_runtimes = {}
        for device in self.device_threads:
            for thread in self.device_threads[device]:
                for function, runtime in self.thread_callgraphs[thread].exclusive_runtime(start=0).items():
                    if device not in exclusive_runtimes:
                        exclusive_runtimes[device] = {}
                    if function not in exclusive_runtimes[device]:
                        exclusive_runtimes[device][function] = 0.0
                    exclusive_runtimes[device][function] += runtime
        
        device_max_sample = {
            device: max(samples, key=lambda m: m.value)
            for device, samples in self.device_samples.items()
        }
        
        for device, attributions in self.function_attributions.items():
            print(f"Device: {device}")
            print(f"   Max Recorded Sample for {device}: {device_max_sample[device] if device in device_max_sample else 'No samples'}")
            for function, metric in sorted(attributions.items(), key=lambda x: x[1].value, reverse=True):
                print(f"   {metric}, {function[:30]}, {exclusive_runtimes[device].get(function, 0.0)}s")
            total_metrics = {}
            for function, metric in attributions.items():
                if total_metrics.get(metric.name) is None and function != "idle":
                    total_metrics[metric.name] = metric
                elif function != "idle":
                    total_metrics[metric.name] = total_metrics[metric.name].add(metric)
            print(f"   Total Attributed Metrics for {device}:")
            for name, total_value in total_metrics.items():
                print(f"      {total_value}, Metric: {name}")
            # Subtract the total value from the max sample to get the idle time
            if device in device_max_sample:
                idle_value = device_max_sample[device].value - sum(m.value for m in attributions.values())
                print(f"   Idle attribution for {device}: {idle_value:.6f} {device_max_sample[device].unit if device_max_sample[device] else ''}")


class TestInterval(unittest.TestCase):
    def test_is_active(self):
        interval = Interval(start=10, end=20)
        self.assertTrue(interval.is_active(15))
        self.assertFalse(interval.is_active(5))
        self.assertFalse(interval.is_active(25))
    
    def test_has_overlap(self):
        interval1 = Interval(start=10, end=20)
        interval2 = Interval(start=15, end=25)
        interval3 = Interval(start=20, end=30)
        
        self.assertTrue(interval1.has_overlap(interval2))
        self.assertFalse(interval1.has_overlap(interval3))
        self.assertTrue(interval2.has_overlap(interval3))
        
    def test_has_overlap_exact_boundary(self):
        interval1 = Interval(start=10, end=20)
        interval2 = Interval(start=20, end=30)
        
        self.assertFalse(interval1.has_overlap(interval2))
        
    def test_is_active_no_end(self):
        interval = Interval(start=10)
        self.assertTrue(interval.is_active(15))
        self.assertTrue(interval.is_active(25))
        self.assertFalse(interval.is_active(5))
        
    def test_unpacked_interval(self):
        interval = Interval(start=10, end=20, depth=1, name="Test", metadata={"key": "value"})
        self.assertEqual(interval.start, 10)
        self.assertEqual(interval.end, 20)
        self.assertEqual(interval.depth, 1)
        self.assertEqual(interval.name, "Test")
        self.assertEqual(interval.metadata, {"key": "value"})
        start, end, depth, name, metadata = interval
        self.assertEqual(start, 10)
        self.assertEqual(end, 20)
        self.assertEqual(depth, 1)
        self.assertEqual(name, "Test")
        self.assertEqual(metadata, {"key": "value"})
        
    def test_clip(self):
        interval = Interval(start=10, end=20)
        clipped = interval.clip(12, 18)
        self.assertEqual(clipped.start, 12)
        self.assertEqual(clipped.end, 18)
        
        # Test clipping to the same start and end
        with self.assertRaises(ValueError):
            interval.clip(15, 15)
            
        # Test clipping with no end
        interval_no_end = Interval(start=10)
        clipped_no_end = interval_no_end.clip(12, 18)
        self.assertEqual(clipped_no_end.start, 12)
        self.assertEqual(clipped_no_end.end, 18)
        
        # Test clipping with an invalid range
        with self.assertRaises(ValueError):
            interval.clip(8, 9)
            
        with self.assertRaises(ValueError):
            interval.clip(21, 22)

class TestTimeline(unittest.TestCase):
    def test_enter_and_leave(self):
        timeline = Timeline()
        interval = timeline.enter(start=10, name="Test")
        self.assertEqual(len(timeline.live_intervals), 1)
        self.assertEqual(timeline.live_intervals[0], interval)
        
        timeline.leave(end=20)
        self.assertEqual(len(timeline.finished_intervals), 1)
        self.assertEqual(timeline.finished_intervals[0], interval._replace(end=20))
        self.assertEqual(len(timeline.live_intervals), 0)
    
    def test_get_intervals_between(self):
        timeline = Timeline()
        timeline.enter(start=10, name="Test1")
        timeline.enter(start=15, name="Test2")
        timeline.leave(end=20)
        # Get intervals between 12 and 18
        intervals = timeline.get_intervals_between(start=12, end=18)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0], Interval(start=12, end=18, depth=1, name="Test1", metadata=None))
        self.assertEqual(intervals[1], Interval(start=15, end=18, depth=2, name="Test2", metadata=None))


class TestMetric(unittest.TestCase):
    def test_attribute(self):
        metric = Metric(name="Energy", value=100, time=10)
        attributed_metric = metric.attribute(0.5)
        self.assertEqual(attributed_metric.value, 50)
        
    def test_add(self):
        metric1 = Metric(name="Energy", value=100, time=10)
        metric2 = Metric(name="Energy", value=50, time=15)
        combined_metric = metric1.add(metric2)
        self.assertEqual(combined_metric.value, 150)
        
    def test_add_different_names(self):
        metric1 = Metric(name="Energy", value=100, time=10)
        metric2 = Metric(name="Power", value=50, time=15)
        with self.assertRaises(ValueError):
            metric1.add(metric2)
            
class TestMetricAttribution(unittest.TestCase):
    def test_single_device_sample(self):
        attribution = MetricAttribution({"CPU": ["Master Thread"]})
        # init_metric = Metric(name="Energy", value=0, time=0, device="CPU", thread="Master Thread")
        attribution.initialize_metrics(Metric(name="Energy", value=0, time=0, device="CPU", thread="Master Thread"), devices=["CPU"])
        metric1 = Metric(name="Energy", value=100, time=15, device="CPU", thread="Master Thread")
        metric2 = Metric(name="Energy", value=200, time=20, device="CPU", thread="Master Thread")
        
        # attribution.sample(init_metric)
        attribution.enter(time=0, thread="Master Thread", function="FunctionA")
        attribution.leave(time=15, thread="Master Thread")
        attribution.sample(metric1)
        attribution.enter(time=15, thread="Master Thread", function="FunctionB")
        attribution.leave(time=20, thread="Master Thread")
        attribution.sample(metric2)
        
        # Report should show the attributed metrics
        attribution.report()
        
        # Check if the function attributions are correct
        self.assertIn("FunctionA", attribution.function_attributions["CPU"])
        self.assertIn("FunctionB", attribution.function_attributions["CPU"])
        self.assertEqual(attribution.function_attributions["CPU"]["FunctionA"].value, 100)
        self.assertEqual(attribution.function_attributions["CPU"]["FunctionB"].value, 200)
        
    def test_multiple_devices(self):
        attribution = MetricAttribution({"CPU": ["Master Thread"], "GPU": ["Worker Thread"]})
        attribution.initialize_metrics(Metric(name="Energy", value=0, time=0, device="CPU", thread="Master Thread"), devices=["CPU"])
        attribution.initialize_metrics(Metric(name="Energy", value=0, time=0, device="GPU", thread="Worker Thread"), devices=["GPU"])
        
        metric1 = Metric(name="Energy", value=100, time=15, device="CPU", thread="Master Thread")
        metric2 = Metric(name="Energy", value=200, time=20, device="GPU", thread="Worker Thread")
        
        attribution.enter(time=0, thread="Master Thread", function="FunctionA")
        attribution.leave(time=15, thread="Master Thread")
        attribution.sample(metric1)
        
        attribution.sample(Metric(name="Energy", value=0, time=15, device="GPU", thread="Worker Thread"))  # Initialize GPU metric
        attribution.enter(time=15, thread="Worker Thread", function="FunctionB")
        attribution.leave(time=20, thread="Worker Thread")
        attribution.sample(metric2)
        
        # Check if the function attributions are correct
        self.assertIn("FunctionA", attribution.function_attributions["CPU"])
        self.assertIn("FunctionB", attribution.function_attributions["GPU"])
        self.assertEqual(attribution.function_attributions["CPU"]["FunctionA"].value, 100)
        self.assertEqual(attribution.function_attributions["GPU"]["FunctionB"].value, 200)

if __name__ == '__main__':
    unittest.main()