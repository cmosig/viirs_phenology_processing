# Land-surface phenology composite

A 10-year, day-of-year composite of the growing season, aggregated from 500 m
satellite phenology to 10 km and 40 km cells, plus the season start, end and
middle derived from it. The middle date is what the deadtrees inference pipeline
uses to decide which Sentinel-2 composite a block is predicted for.

## Source

`0_download.sh` pulls **VNP22Q2.001** — the VIIRS Global Land Surface Phenology
yearly product, 500 m, two cycles per year, on the MODIS sinusoidal grid. Despite
the file and variable names throughout this repo, the input is VIIRS, not MODIS.
Years currently held: 2013-2022 (10 years x 2 cycles = 20 layers).

Two layers drive everything downstream:

- `Onset_Greenness_Maximum` — the date the canopy reaches full greenness
  (**maturity**, not leaf-out)
- `Onset_Greenness_Decrease` — the date greenness starts to fall
  (**senescence onset**, not leaf-off)

Both are stored as *days since 2000-01-01*, so turning them into a day of year
means subtracting the real day count to 1 January of that layer's year.

The product carries more that this pipeline currently ignores:
`EVI2_Growing_Season_Area`, `EVI2_Onset_Greenness_Maximum`,
`Growing_Season_Length`, `Greenness_Agreement_Growing_Season`, the per-phase
quality flags `PGQ_*` and `GLSP_QC`, and the mid-phase dates. See
[Known limitations](#known-limitations).

## Pipeline

| step | script | in | out |
|---|---|---|---|
| download | `0_download.sh` | LP DAAC | HDF tiles |
| to zarr | `1_convert_to_zarr.py` | HDF tiles | `modispheno.zarr` (20, 33600, 86400) |
| forest mask | `2_aggregate_world_cover.py` | WorldCover | `worldcover_aggregated.tif` (500 m tree-cover fraction) |
| composite | `3_create_phenology_aggregate.py` | the two above | `modispheno_aggregated_v6.zarr` (1680, 4320, 366) |
| dates | `4_aggregate_to_dates.py` | composite | `modis_pheno_processed_v6{,_10km}.zarr` |
| normalise | `SIDE_normalize_curve.py` | composite | `..._v6_normalized.zarr` (uint8 0-255 + `nan_mask`) |
| gap fill | `fill_modis_phenology_nn.py` (in `sentinel_mortality/scripts/misc/`) | normalised | `..._v6_normalized_filled.zarr` |
| GeoTIFF | `export_geotiff.py` | any dates store | 6-band `.tif` |
| QGIS | `make_qgis_project.py` | the tifs | `qgis/phenology.qgs` |

### The composite (step 3)

For every 10 km cell (20x20 pixels of 500 m) and every day of year, count how
many forested pixel-years had that day inside a growing season. A pixel-year
contributes the interval `[maturity, senescence onset)`, with the documented
fallbacks: senescence missing -> the season runs to day 366; senescence present
but before maturity, or maturity missing -> the season runs from day 0. The two
cycles of one pixel-year are a **union**, not a sum. A season crossing 31
December wraps onto the start of the same composite year, because the composite
is a day-of-year histogram and therefore cyclic. Only pixels the WorldCover mask
calls forest contribute; a cell with no forested pixel is left empty.

The cube's chunks are `(1, 2400, 2400)`, so the composite is built one source
chunk at a time — 120x120 cells, every chunk decompressed exactly once — and
each tile is written as a single output chunk. The interval marking is a
difference array plus one cumulative sum, never a per-day write. A global run is
~4 minutes on 48 workers.

### The dates (step 4)

The 10 km composite is optionally block-summed to 40 km (`--factor 4`, the
default; `--factor 1` keeps 10 km), then per cell:

1. `middle` = the day the most pixel-years are in leaf. Flat-topped curves have
   many days tied at the maximum, so it is the midpoint of the contiguous
   plateau of tied days holding the peak — still a day at the maximum, chosen
   deterministically rather than by which tied day comes first.
2. threshold the count curve at 50% of its own min-max range and keep the run
   **containing that peak**
3. `start` = first day of that run, `end` = last

The date exists to choose the Sentinel-2 composite on which a leafless crown
most likely means a dead tree, so it belongs where a healthy tree is most
certainly in leaf. Through v5 `middle` was instead the midpoint between `start`
and `end`, and the run was chosen by length; see [Versions](#versions).

Because the plateau lies inside the run, `middle` is always within
`[start, end]` — checked over the whole grid, measured and interpolated, at both
resolutions.

Each of the three is written twice: the raw band (`start`, `end`, `middle`),
which is NaN wherever no cycle could be derived, and an `_interp` band, which is
the same values nearest-valid-filled everywhere. Inference reads
`middle_interp`. The fill takes the whole triple from one donor cell: filling the
three independently let a cell draw its start from one neighbour and its middle
from another, producing a season nobody measured — which matters because
`extend_metadata.py` tests `start <= acquisition <= end` against these bands to
decide which orthophotos enter training.

## What `start`, `end` and `middle` mean

They are *not* greenup and leaf-off. `start` is when the canopy reaches full
greenness and `end` is when it starts to decline, so the pair brackets the
full-canopy plateau and `middle` sits inside it. Compared against PhenoCam
GCC transitions at two deciduous sites:

| | camera rising 50% | our `start` | our `middle` | our `end` | camera falling 50% |
|---|---|---|---|---|---|
| Harvard Forest | 132 | 171 | 205 | 245 | 286 |
| Bartlett | 138 | 175 | 209 | 244 | 266 |

## Versions

Every version is the same recipe with a defect removed. Differences are measured
as the shift in mid-season day of year at 170,707 inference-block centroids.

| step | what changed | median | >30 d |
|---|---|---|---|
| v2 -> v3 | `nansum` when block-summing (a 40 km cell no longer goes empty because one of its 16 sub-cells is), and a min-max rather than 0-max threshold | 2 d | 17% |
| v3 -> v4 | day-of-year conversion used `366 * (year - 2000)`, over-subtracting by 9 d in 2013 growing to 16 d in 2022; cycle 2 read cycle 1's greenup as its start; the forest mask was read one cell too far south | **12 d** | **19%** |
| v4 -> v5 | a season crossing New Year is wrapped rather than clipped at day 366 | 0 d | 0.4% |
| v5 -> v6 | `middle` moves to the peak of the curve, and the kept run is the one containing that peak; the whole triple is filled from one donor | 4 d | 10% |

v4 and v5 were measurements, not products: only v6 is kept, and every store
carries that version. v3 and v5 changed the composite, v6 only step 4, so the
composite `modispheno_aggregated_v6.zarr` is what v5 produced. The year-offset
fix moves the product onto the source dates: central Germany's median source
maturity is DOY 198, v2/v3 said 181, v6 says 198. Across six flux and phenology
sites, `start` and `end` reproduce the median source maturity and senescence
within 0-3 days.

### Why the peak (v6)

Scored by *P(in leaf)* — the share of pixel-years in season on the chosen day,
relative to the best day of that cell — over the Amazon, where the curve has the
least shape to work with:

| design | P(in leaf) | cells below 0.8 | neighbour dev | >30 d |
|---|---|---|---|---|
| midpoint of the 50% run (v5) | 0.919 | 20.3% | 11.7 d | 30.6% |
| **peak of the curve (v6)** | **1.000** | **0.0%** | 8.3 d | 17.3% |
| circular mean of the curve | 0.940 | 7.7% | 6.1 d | 9.8% |
| centre of the >=90% plateau | 0.982 | 0.0% | 6.3 d | 15.6% |

The v5 midpoint lands below 80% of peak leaf-on in a fifth of Amazon cells. The
circular mean is the spatially smoothest but in bimodal cells falls *between* the
two seasons. The peak maximises the objective by construction.

Keeping the run that contains the peak matters: chosen by length, the run would
not contain the peak in 8% of Amazon cells (2.2% SE Asia, 0% Europe), which would
put `middle` outside `[start, end]`. It also replaces the length tie-break
between two similar cycles with a meaningful rule -- the season holding the
most-in-leaf day is the main season. Selecting instead by integrated count, peak
height or prominence changes at most 10% of decisions and improves nothing.

Effect of v6 at the 170,707 inference-block centroids: median shift 4 d overall,
1 d north of 45N (0% beyond 30 d), 6 d in 0-45N and 8 d south of the equator.
Globally the share of measured cells whose date sits below 80% of peak leaf-on
goes from 0.2% to 0.0%, and in the Amazon from 20.3% to 0%.

## The sample-size gate

A 10 km cell holds at most 400 forested pixels x 10 years. Coastal and
fragmented cells hold a handful, and a date read off a curve built from a
handful of pixel-years is unstable — these were the isolated specks along
coastlines.

The threshold is measured. Taking well-sampled cells (>=2000 pixel-years; 120
cells over 4 tiles), drawing random subsets of *k* pixel-years, extracting the
date from each subset and comparing against that same cell's full-sample date:

| pixel-years | 5 | 20 | 50 | 100 | **200** | 400 | 800 |
|---|---|---|---|---|---|---|---|
| 95th pct error | 82 d | 27 d | 13 d | 8 d | **5 d** | 3 d | 2 d |

The date is only ever used to pick a **7-day** composite, so the cut goes where
the sampling error drops below that step: **200 pixel-years**
(`--min-pixel-years`). Below it the cell is left to the interpolation, exactly
like a cell with no measured cycle. This drops 11.8% of measured cells at 40 km
and 17.8% at 10 km, and takes the share of cells disagreeing with their
neighbours by more than 30 days from 6.2% to 3.3% on a Mediterranean box.
Smoothing the curve instead was tried and does almost nothing (7.4% -> 7.1%),
which is what pointed at sample size rather than high-frequency noise.

## Known limitations

- **Weak seasonality breaks the threshold.** The date comes from a 50%-of-range
  crossing, so it needs the curve to have a range. In central Europe the curve
  falls to zero in winter (contrast 1.00 in every cell) and neighbouring cells
  agree to ~2 days. In the Amazon, seasons are asynchronous between pixels, the
  curve never returns to zero, and the crossing is decided by small fluctuations:

  | seasonal contrast `(max-min)/max` | share of Amazon cells | >30 d neighbour jump |
  |---|---|---|
  | < 0.25 | 0.4% | 45% |
  | 0.25-0.50 | 28% | 40% |
  | 0.50-0.75 | 34% | 22% |
  | > 0.75 | 37% | 2.4% |

  Amazon cells with strong contrast are as clean as European ones. The residual
  noise there is a property of the signal, not of the sampling — those cells are
  already at the maximum pixel count.

- **The two-cycle choice.** v6 keeps the cycle holding the peak, which is
  principled but still uses only the two date layers. The discriminating
  information sits in the source: `EVI2_Growing_Season_Area` differs between a
  pixel-year's two cycles by a median factor of 1.7, and only 21% of two-cycle
  pixel-years are within 25%. Weighting each pixel-year by its growing-season
  area, or dropping the ones `PGQ_*` flags as poor, would let the dominant season
  win by mass before any selection happens. Not done.

- **`middle_interp` hides how much is measured.** 43% of inference blocks sit on
  a cell with no measured cycle at all, filled from the nearest valid cell.
  Always check the raw band before trusting a value.

- **Day of year is cyclic.** Rendering it with a non-cyclic ramp tears the map at
  the year boundary and invents structure — 34% of the Amazon lies within 45 days
  of New Year. `qgis/phenology.qgs` uses a cyclic ramp over day 1-366.

## Running it

Only `modispheno_aggregated_v6_normalized_filled.zarr` is kept zipped, since that
is the one that gets shipped; everything else is read in place.

```bash
python 3_create_phenology_aggregate.py --workers 48            # ~4 min, ~25 GB
python 4_aggregate_to_dates.py                                 # 40 km
python 4_aggregate_to_dates.py --factor 1                      # 10 km
#   --middle threshold reproduces the pre-v6 rule
python SIDE_normalize_curve.py                                 # normalised curve
python export_geotiff.py --store modis_pheno_processed_v6.zarr
QT_QPA_PLATFORM=offscreen /usr/bin/python3 make_qgis_project.py # system python has PyQGIS
```

Paths come from `paths.py` (`DATAPATH`). The QGIS project keeps its GeoTIFFs in
`qgis/` so a clone opens anywhere.
