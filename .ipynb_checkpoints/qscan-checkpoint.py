#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import os

# CONFIG
csv_file = "/home/charliewilliam.winborn/projects/glitch/omicron_project/temp_csvs/table-LSC_POP_A_LF_OUT_DQ-snr_40d0-start_1437436820-end_1437523218.csv"
channel = "H1:LSC-POP_A_LF_OUT_DQ"
outdir = "./qscans/"
pad = 1.0   # seconds before and after

os.makedirs(outdir, exist_ok=True)

# Load triggers
df = pd.read_csv(csv_file)

for i, row in df.iterrows():
    gps_time = float(row["time"])
    start = gps_time - pad
    end   = gps_time + pad

    print(f"[{i}] Fetching data {start} → {end} around trigger {gps_time}")

    try:
        ts = TimeSeries.fetch(channel, start, end)
    except Exception as e:
        print(f"   Could not fetch data: {e}")
        continue

    # Q-transform (q-scan)
    qscan = ts.q_transform(outseg=(start, end),
                           qrange=(4, 16), frange=(10, 100),
                           logf=True)

    # Plot
    plot = qscan.plot()
    ax = plot.gca()
    ax.set_title(f"Q-scan around GPS {gps_time}")
    ax.legend()

    # Save
    outfile = os.path.join(outdir, f"qscan_{i}_{int(gps_time)}.png")
    plot.savefig(outfile)
    plt.close(plot.figure)

