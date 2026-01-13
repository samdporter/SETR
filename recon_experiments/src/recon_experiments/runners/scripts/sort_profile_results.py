import argparse
import pstats


def print_leaf_hotspots(profile_path: str, limit: int = 50):
    """
    Print the top `limit` leaf functions (no callees) by self time (tottime).
    """
    s = pstats.Stats(profile_path)
    s.strip_dirs()
    # Collect every function that is ever a *caller* of someone else.
    caller_funcs = set()
    for f, (_, _, _, _, callers_dict) in s.stats.items():
        caller_funcs.update(callers_dict.keys())
    # Leaves: functions that never appear as a caller.
    leaves = []
    leaves.extend(
        (tottime, ncalls, f)
        for f, (_, ncalls, tottime, _, _callers) in s.stats.items()
        if f not in caller_funcs
    )
    leaves.sort(key=lambda x: x[0], reverse=True)
    for tottime, ncalls, f in leaves[:limit]:
        print(f"{tottime:10.3f}s  ncalls={ncalls:<9}  {pstats.func_std_string(f)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", help="Path to the profile file")
    parser.add_argument("--limit", type=int, default=50, help="Number of hotspots to display")
    return parser.parse_args()


def main(args):
    print_leaf_hotspots(args.profile, args.limit)


if __name__ == "__main__":
    args = parse_args()
    main(args)
