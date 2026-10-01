#!/usr/bin/env python3
# Hold N GB on a GPU so a server sees a smaller card (12 GB simulation on the 3090).
import sys, time, torch
gb = float(sys.argv[1]); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
x = torch.empty(int(gb * (1 << 30)), dtype=torch.uint8, device=dev)
free, total = torch.cuda.mem_get_info(torch.device(dev))
print(f"hog: holding {gb:.2f} GB on {dev}; free now {free/1e9:.2f} GB of {total/1e9:.2f}", flush=True)
while True: time.sleep(3600)
