import json
import numpy as np

json_path = "/Users/aryahb/Downloads/raw_videos.json"

with open(json_path, "r", encoding="utf-8") as f:
    videos = json.load(f)

durations = [
    v["duration"]
    for v in videos
    if isinstance(v.get("duration"), (int, float))
]

durations = np.array(durations)

percentiles = [25, 50, 75, 80, 90, 95, 96, 97, 98, 99, 100]

print(f"Total videos with duration: {len(durations)}")
print()

print("Duration percentile summary:")
for p in percentiles:
    cutoff = np.percentile(durations, p)

    count_at_or_below = np.sum(durations <= cutoff)
    count_above = np.sum(durations > cutoff)

    print(f"{p}% percentile:")
    print(f"  duration cutoff: {cutoff:.2f} seconds")
    print(f"  videos <= cutoff: {count_at_or_below}")
    print(f"  videos > cutoff:  {count_above}")
    print()