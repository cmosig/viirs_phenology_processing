"""Compare dates stores: year-boundary artifacts, coverage, and where dates move.

Every version of this pipeline is the same recipe with a defect removed, so the
question is always the same: did the artifact go, what did it cost in coverage,
and which regions actually moved.

    python compare_versions.py v6 v7 v8 v9
"""

import argparse
from os.path import join

import numpy as np
import xarray as xr
from paths import DATAPATH

EDGE = 10  # days either side of 1 January counted as "at the year boundary"


def load(version, res):
    suffix = "" if res == 40 else "_10km"
    return xr.open_zarr(
        join(DATAPATH, f"modis_pheno_processed_{version}{suffix}.zarr")
    ).phenology40km.load()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("versions", nargs="+")
    ap.add_argument("--res", type=int, default=10, choices=(10, 40))
    args = ap.parse_args()

    stores = {v: load(v, args.res) for v in args.versions}

    print(f"{args.res} km stores\n")
    print(f"{'version':8s} {'measured cells':>15s} {'modal day':>10s} {'x mean':>7s} "
          f"{'day 0':>8s} {'day 365':>8s} {'within 10 d of 1 Jan':>21s}")
    for v, da in stores.items():
        m = da.sel(var="middle").values
        meas = ~np.isnan(m)
        vals = np.round(m[meas]).astype(int)
        h = np.bincount(vals, minlength=367)[:366]
        mean = h.mean()
        near = ((vals <= EDGE) | (vals >= 366 - EDGE)).mean()
        # expected share of a 21-day band if dates were spread evenly
        expected = (2 * EDGE + 1) / 366
        print(f"{v:8s} {meas.sum():15,} {int(h.argmax()):10d} {h.max()/mean:7.1f} "
              f"{h[0]:8,} {h[365]:8,} {100*near:8.2f}% (even: {100*expected:.2f}%)")

    if len(args.versions) < 2:
        return
    print("\nshift in middle_interp between consecutive versions, by latitude band")
    y = stores[args.versions[0]].y.values
    # sinusoidal: y is metres north, radius 6371007.181
    lat = np.degrees(y / 6371007.181)
    bands = [("tropics |lat|<23.5", np.abs(lat) < 23.5),
             ("mid 23.5-50", (np.abs(lat) >= 23.5) & (np.abs(lat) < 50)),
             ("high >=50", np.abs(lat) >= 50)]
    for a, b in zip(args.versions, args.versions[1:]):
        A = stores[a].sel(var="middle_interp").values
        B = stores[b].sel(var="middle_interp").values
        d = np.abs((B - A + 183) % 366 - 183)
        print(f"  {a} -> {b}")
        for name, rows in bands:
            dd = d[rows]
            print(f"    {name:20s} median {np.median(dd):5.1f} d   "
                  f">30 d {100*(dd > 30).mean():5.1f}%   >90 d {100*(dd > 90).mean():5.1f}%")


if __name__ == "__main__":
    main()
