#!/usr/bin/env python3
"""
Generate a Weather Underground–style precipitation chart for San Diego (KSAN area),
including historical climatology bands and current-year overlay updated daily.

Key idea for centered 31-day smoothing near "today":
- Solid line: centered 31-day mean only where full window is known (up to last_actual_date - 15 days)
- Dashed tail: fill missing future days in the half-window with climatology mean (so the curve can extend to today)

Data source: Meteostat (aggregates public sources incl. NOAA). See docs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, date, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from meteostat import Point, Daily


INCH_PER_MM = 0.03937007874

# Approx KSAN (San Diego Intl). Using point lookup keeps it robust even if station IDs change.
KSAN_POINT = Point(32.7338, -117.1933)

# WMO-ish standard climatology period. You can change if you prefer "last 30 years".
CLIMO_START_YEAR = 1991
CLIMO_END_YEAR = 2020


def md_index_365() -> list[str]:
    """Return month-day strings for a non-leap year (365 entries)."""
    base = pd.date_range("2001-01-01", "2001-12-31", freq="D")  # 2001 is non-leap
    return [d.strftime("%m-%d") for d in base]


def drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    mask = ~((idx.month == 2) & (idx.day == 29))
    return df.loc[mask].copy()


def centered_31d(series: pd.Series) -> pd.Series:
    """Centered 31-day mean. Requires full window."""
    return series.rolling(31, center=True, min_periods=31).mean()


@dataclass
class Bands:
    p10: np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p90: np.ndarray
    # also keep daily mean for tail fill:
    daily_mean: np.ndarray
    daily_p25: np.ndarray
    daily_p75: np.ndarray


def fetch_daily_prcp(point: Point, start: date, end: date) -> pd.Series:
    """
    Fetch daily precip in inches from Meteostat for [start, end].
    Meteostat daily prcp is in mm.
    """

    # --- FIX: Meteostat expects datetime, not date ---
    if isinstance(start, date) and not isinstance(start, datetime):
        start = datetime.combine(start, time.min)
    if isinstance(end, date) and not isinstance(end, datetime):
        end = datetime.combine(end, time.min)
    # -----------------------------------------------

    df = Daily(point, start, end).fetch()
    if df.empty or "prcp" not in df.columns:
        raise RuntimeError("No precipitation data returned from Meteostat for requested range.")
    df = df[["prcp"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    pr = df["prcp"].astype(float) * INCH_PER_MM
    pr = pr.fillna(0.0)
    return pr


def build_climatology_bands(prcp_in: pd.Series) -> Bands:
    """
    prcp_in: daily precip in inches over CLIMO_START_YEAR..CLIMO_END_YEAR inclusive
    Returns bands for the centered 31-day mean per month-day (365-day calendar, Feb 29 removed).
    """
    # Build daily climatology too (for tail fill)
    df = prcp_in.to_frame("prcp")
    df = drop_feb29(df)
    df["year"] = df.index.year
    df["md"] = df.index.strftime("%m-%d")

    md_list = md_index_365()

    # Daily (unsmoothed) climatology distribution by month-day
    daily_groups = df.groupby("md")["prcp"]
    daily_mean = np.array([daily_groups.mean().get(md, 0.0) for md in md_list])
    daily_p25 = np.array([daily_groups.quantile(0.25).get(md, 0.0) for md in md_list])
    daily_p75 = np.array([daily_groups.quantile(0.75).get(md, 0.0) for md in md_list])

    # For smoothed bands, compute centered 31d mean within each year, then gather across years per md
    per_year_smoothed = []
    for y in range(CLIMO_START_YEAR, CLIMO_END_YEAR + 1):
        s = df.loc[df["year"] == y, "prcp"].copy()
        # Ensure full date coverage within year to make rolling stable
        yr_start = pd.Timestamp(f"{y}-01-01")
        yr_end = pd.Timestamp(f"{y}-12-31")
        full = pd.date_range(yr_start, yr_end, freq="D")
        tmp = s.reindex(full).fillna(0.0).to_frame("prcp")
        tmp = drop_feb29(tmp)
        sm = centered_31d(tmp["prcp"])
        tmp = tmp.assign(sm=sm, md=tmp.index.strftime("%m-%d"))
        per_year_smoothed.append(tmp[["md", "sm"]])

    sm_all = pd.concat(per_year_smoothed, ignore_index=True)
    sm_groups = sm_all.groupby("md")["sm"]

    def q_arr(q: float) -> np.ndarray:
        return np.array([sm_groups.quantile(q).get(md, np.nan) for md in md_list])

    p10 = q_arr(0.10)
    p25 = q_arr(0.25)
    p50 = q_arr(0.50)
    p75 = q_arr(0.75)
    p90 = q_arr(0.90)

    # Fill edge NaNs (from min_periods=31) by nearest valid values for a smooth loop.
    # This affects early Jan / late Dec slightly in the climatology display.
    def fill_edges(a: np.ndarray) -> np.ndarray:
        s = pd.Series(a).interpolate(limit_direction="both").to_numpy()
        return s

    return Bands(
        p10=fill_edges(p10),
        p25=fill_edges(p25),
        p50=fill_edges(p50),
        p75=fill_edges(p75),
        p90=fill_edges(p90),
        daily_mean=daily_mean,
        daily_p25=daily_p25,
        daily_p75=daily_p75,
    )


def make_current_year_series(prcp_y: pd.Series, bands: Bands) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      solid_sm (length 365, NaN where not defined)
      dashed_sm (length 365, NaN where not defined)
      last_known_md_index (int): last index where solid is valid
    """
    df = prcp_y.to_frame("prcp")
    df = drop_feb29(df)

    # Reindex to full year calendar through today (or last data day)
    y = df.index.year.max()
    year_start = pd.Timestamp(f"{y}-01-01")
    year_end = pd.Timestamp(f"{y}-12-31")
    full = pd.date_range(year_start, year_end, freq="D")
    full = full[~((full.month == 2) & (full.day == 29))]  # keep 365
    df = df.reindex(full).fillna(0.0)

    # Determine last actual observation day we fetched (usually yesterday/today depending on source)
    last_actual = prcp_y.index.max()
    if isinstance(last_actual, pd.Timestamp):
        last_actual = last_actual.to_pydatetime().date()

    # Build a 15-day climatology extension beyond last_actual so centered rolling can be computed for the tail
    md_list = md_index_365()
    md_to_climo = {md: bands.daily_mean[i] for i, md in enumerate(md_list)}

    tail_days = 15
    ext_index = pd.date_range(full.min(), full.max() + pd.Timedelta(days=tail_days), freq="D")
    ext_index = ext_index[~((ext_index.month == 2) & (ext_index.day == 29))]
    ext = pd.Series(index=ext_index, dtype=float)

    # Fill known within-year days from df, and fill future extension with daily climatology mean
    for ts in ext_index:
        d = ts.date()
        if d <= last_actual:
            # within fetched range (we reindexed missing to 0)
            if ts in df.index:
                ext.loc[ts] = float(df.loc[ts, "prcp"])
            else:
                ext.loc[ts] = 0.0
        else:
            md = ts.strftime("%m-%d")
            ext.loc[ts] = float(md_to_climo.get(md, 0.0))

    sm_ext = centered_31d(ext)

    # Extract smoothed values for the 365-day year
    sm_year = sm_ext.reindex(full).to_numpy()

    # Solid segment ends at last_actual - 15 days (because centered needs +15 days actual)
    solid_end_date = last_actual - timedelta(days=15)
    solid_end_ts = pd.Timestamp(solid_end_date)

    solid = sm_year.copy()
    dashed = sm_year.copy()

    # Where full centered window is not fully observed, make solid NaN and keep dashed
    solid_mask = full <= solid_end_ts
    solid[~solid_mask] = np.nan

    # For dashed, only show from (solid_end+1) through last_actual (today-ish). Hide rest of year.
    dashed_mask = (full > solid_end_ts) & (full <= pd.Timestamp(last_actual))
    dashed[~dashed_mask] = np.nan

    last_known_idx = int(np.where(solid_mask)[0].max()) if solid_mask.any() else -1
    return solid, dashed, last_known_idx


def plot_chart(out_png: str, out_meta: str, bands: Bands, solid: np.ndarray, dashed: np.ndarray, updated_str: str):
    md_list = md_index_365()
    x = np.arange(len(md_list))

    fig = plt.figure(figsize=(14, 4), dpi=180)
    ax = plt.gca()

    # Historical bands
    ax.fill_between(x, bands.p10, bands.p90, alpha=0.15)
    ax.fill_between(x, bands.p25, bands.p75, alpha=0.25)
    ax.plot(x, bands.p50, linewidth=2)

    # Current year overlay
    ax.plot(x, solid, linewidth=2.2)
    ax.plot(x, dashed, linewidth=2.0, linestyle="--")

    # Axes formatting (match the clean WU style)
    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(bottom=0)

    # Month ticks
    months = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
    month_pos = [(m - pd.Timestamp("2001-01-01")).days for m in months]
    month_lbl = [m.strftime("%b") for m in months]
    ax.set_xticks(month_pos)
    ax.set_xticklabels(month_lbl)

    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.set_ylabel("Precip (in) — centered 31-day mean")

    ax.set_title("San Diego (KSAN area) — daily precipitation, smoothed (WU-style)")

    # Small annotation
    ax.text(
        0.995, 0.02,
        f"Updated: {updated_str}\nBands: {CLIMO_START_YEAR}–{CLIMO_END_YEAR}\nDashed tail uses climatology for future half-window",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8, alpha=0.8
    )

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Write a tiny metadata file for the webpage
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write(updated_str.strip() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="site", help="Output directory for site assets")
    args = ap.parse_args()

    outdir = args.outdir.rstrip("/")

    today = date.today()
    # Pull a little buffer into the current year so smoothing has context, and include yesterday/today
    start_all = date(CLIMO_START_YEAR, 1, 1)
    end_all = today

    pr = fetch_daily_prcp(KSAN_POINT, start_all, end_all)

    # Build bands from climatology window
    pr_climo = pr.loc[f"{CLIMO_START_YEAR}-01-01": f"{CLIMO_END_YEAR}-12-31"]
    bands = build_climatology_bands(pr_climo)

    # Current year series
    cy = today.year
    pr_y = pr.loc[f"{cy}-01-01": str(today)]
    solid, dashed, _ = make_current_year_series(pr_y, bands)

    # Output paths
    import os
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, "sd-precip.png")
    out_meta = os.path.join(outdir, "last-updated.txt")

    updated_str = datetime.now().strftime("%Y-%m-%d %H:%M (local build time)")
    plot_chart(out_png, out_meta, bands, solid, dashed, updated_str)


if __name__ == "__main__":
    main()
