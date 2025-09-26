import argparse, itertools, math

def compute_n_steps(target_seconds, ta_s, ts_s, method="round", min_steps=1, max_steps=None):
    period = ta_s + ts_s
    if period <= 0:
        return max(1, min_steps)
    raw = target_seconds / period
    if method == "auto":
        # Usa ceil si subestimar te aleja más que sobreestimar (según tolerancia)
        up, down = math.ceil(raw), math.floor(raw)
        return max(min_steps, 1 if max_steps is None else min(up, max_steps))
    n = {"ceil": math.ceil, "floor": math.floor}.get(method, round)(raw)
    n = max(min_steps, int(n))
    if max_steps is not None:
        n = min(n, max_steps)
    return n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--values", nargs="+", type=float,
                   default=[1,2,5,20,50,100,150,200,400,1000],
                   help="Durations in milliseconds.")
    p.add_argument("--target-seconds", type=float, default=30.0)
    p.add_argument("--method", choices=["round","ceil","floor","auto"], default="round")
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--format", choices=["csv","commands"], default="csv")
    p.add_argument("--emit-units", choices=["s","ms"], default="ms",
                   help="Units to print --time_active/--time_sleep. Default: milliseconds.")
    p.add_argument("--unique-pairs", action="store_true",
                   help="Only ta<=ts to avoid symmetric duplicates.")
    p.add_argument("--srun-prefix", default="srun -A gen010 -t10 -N 1")
    p.add_argument("--use-single-gcd", action="store_true",
                   help="Run the benchmark with just one GCD per GPU instead of two.")
    p.add_argument("--bench-cmd", default="$BENCHMARK_EXEC")
    p.add_argument("--vector-size", default="${VECTOR_SIZE}")
    args = p.parse_args()

    if args.use_single_gcd:
        args.srun_prefix +=  ' --ntasks-per-node=4 --gpus-per-task=2'
    else:
        args.srun_prefix +=  ' --ntasks-per-node=8 --gpus-per-task=1'
    args.srun_prefix += ' --gpu-bind=closest'
        # args.srun_prefix = args.srun_prefix.replace('--ntasks-per-node=8 --gpus-per-task=1', '--ntasks-per-node=4 --gpus-per-task=2')
        

    # Internamente trabajamos en segundos
    values_s = [v/1000.0 for v in args.values]

    def pairs():
        it = itertools.product(values_s, values_s)
        return ((ta, ts) for ta, ts in it if not args.unique_pairs or ta <= ts)

    rows = []
    for ta_s, ts_s in pairs():
        n = compute_n_steps(args.target_seconds, ta_s, ts_s,
                            method=args.method, min_steps=args.min_steps, max_steps=args.max_steps)
        approx = n*(ta_s+ts_s)
        rows.append(dict(time_active_s=ta_s, time_sleep_s=ts_s,
                         n_steps=n, approx_duration_s=approx, error_s=approx-args.target_seconds))

    if args.format == "csv":
        print("time_active_s,time_sleep_s,n_steps,approx_duration_s,error_s")
        for r in rows:
            print(f"{r['time_active_s']:.6f},{r['time_sleep_s']:.6f},{r['n_steps']},{r['approx_duration_s']:.6f},{r['error_s']:.6f}")
    else:
        for r in rows:
            # Emisión en las unidades pedidas
            if args.emit_units == "ms":
                ta_out = int(round(r["time_active_s"]*1000))
                ts_out = int(round(r["time_sleep_s"]*1000))
            else:
                ta_out = r["time_active_s"]
                ts_out = r["time_sleep_s"]
            cmd = (
                f"{args.srun_prefix} "
                f"{args.bench_cmd} "
                f"--vector_size {args.vector_size} "
                f"--n_steps {r['n_steps']} "
                f"--time_active {ta_out} "
                f"--time_sleep {ts_out}"
            )
            print(cmd)

if __name__ == "__main__":
    main()