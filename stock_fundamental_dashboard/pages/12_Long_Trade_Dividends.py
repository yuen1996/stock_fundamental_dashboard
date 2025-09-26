# ─────────────────────────────────────────
# Dividend Projection (ENTRY-AWARE, TAX-AWARE)
#   • Only count years after you actually opened the position
#   • Prorate the open year and the current year by holding time
#   • DPS source: master Add/Edit dataset (annual rows only)
#   • Tax: per-stock override via 'TaxPct' column (if present), else default
# ─────────────────────────────────────────

st.markdown(
    '<div class="sec info"><div class="t">⚙️ Dividend settings</div>'
    '<div class="d">Use your entry date, shares, and tax to estimate per-year dividends.</div></div>',
    unsafe_allow_html=True,
)
s1, s2 = st.columns([1, 1])
with s1:
    default_tax_pct = st.number_input("Default Dividend Tax %", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
with s2:
    prorate_mode = st.selectbox("Prorate method", ["Daily (precise)", "Monthly (simple)"], index=0)

# --- helpers -------------------------------------------------------
def _year_bounds(y: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=y, month=1, day=1)
    end   = pd.Timestamp(year=y, month=12, day=31, hour=23, minute=59, second=59)
    return start, end

def _days_in_year(y: int) -> int:
    s, e = _year_bounds(y)
    return (e - s).days + 1

def _months_in_year(y: int) -> int:
    return 12

def _held_fraction_for_year(open_dt: pd.Timestamp, year: int, today: pd.Timestamp, mode: str) -> float:
    """
    Fraction of the year actually held, given the entry date (open_dt).
    For open trades we assume still holding until 'today'.
    """
    y_start, y_end = _year_bounds(year)
    start = max(open_dt.normalize(), y_start)
    end   = min(today.normalize(), y_end)
    if end < y_start or start > y_end or end < start:
        return 0.0

    if mode.startswith("Daily"):
        total = _days_in_year(year)
        held  = (end - start).days + 1
        return max(0.0, min(1.0, held / total))
    else:  # Monthly
        # Count whole months from start.month..end.month within the year
        sm = max(1, start.month) if start.year == year else 1
        em = min(12, end.month)  if end.year == year else 12
        total = _months_in_year(year)
        heldm = max(0, em - sm + 1)
        return max(0.0, min(1.0, heldm / total))

def _pick_dps_col(df: pd.DataFrame) -> str | None:
    # Try the common headers you already use in your dataset
    for c in ["DPS", "Dividend per Share (TTM, RM)", "DPU"]:
        if c in df.columns:
            return c
    # fallback: any column that smells like DPS/DPU
    for c in df.columns:
        cc = str(c).lower()
        if ("dps" in cc) or ("dividend per share" in cc) or ("dpu" in cc):
            return c
    return None

# --- gather inputs from long holdings (with entry and shares) -----
today = pd.Timestamp.now()
lp = filtered[["Name", "Shares", "OpenDate"]].copy()
lp["Shares"]   = pd.to_numeric(lp["Shares"], errors="coerce").fillna(0)
lp["OpenDate"] = pd.to_datetime(lp["OpenDate"], errors="coerce")
# optional per-row tax override if such a column exists in your open positions
if "TaxPct" in filtered.columns:
    lp["TaxPct"] = pd.to_numeric(filtered["TaxPct"], errors="coerce")
else:
    lp["TaxPct"] = None

# --- read annual DPS from master Add/Edit dataset -----------------
src_df = io_helpers.load_data()
dps_by_year = pd.DataFrame(columns=["Year", "Total Dividend (Net RM)"])
breakdown  = pd.DataFrame(columns=["Year", "Name", "Dividend (Net RM)", "Dividend (Gross RM)", "Held %", "Tax %"])

if src_df is not None and not src_df.empty and "Name" in src_df.columns:
    df_all = src_df.copy()
    # annual only
    if "IsQuarter" in df_all.columns:
        df_all = df_all[df_all["IsQuarter"] != True]
    if "Year" in df_all.columns:
        df_all["Year"] = pd.to_numeric(df_all["Year"], errors="coerce")

    dps_col = _pick_dps_col(df_all)
    if dps_col:
        annual = df_all[["Name", "Year", dps_col]].dropna(subset=["Year"]).copy()
        annual["Year"] = pd.to_numeric(annual["Year"], errors="coerce").astype("Int64")
        annual[dps_col] = pd.to_numeric(annual[dps_col], errors="coerce")

        # Merge (inner) to align only stocks we currently hold as Long Trade
        merged = annual.merge(lp, on="Name", how="inner")

        # Compute held fraction per row/year using OpenDate
        merged["Held %"] = merged.apply(
            lambda r: (
                0.0 if pd.isna(r["OpenDate"]) or pd.isna(r["Year"])
                else _held_fraction_for_year(
                    pd.Timestamp(r["OpenDate"]),
                    int(r["Year"]),
                    today,
                    prorate_mode
                ) * 100.0
            ),
            axis=1,
        )

        # Ignore years before entry (Held % = 0) automatically via fraction 0
        merged["Gross"] = (merged[dps_col] * merged["Shares"] * (merged["Held %"] / 100.0)).round(2)

        # Apply per-stock tax override if present, else default page-level tax
        def _tax_pct(row) -> float:
            try:
                v = float(row.get("TaxPct"))
                if np.isfinite(v):
                    return max(0.0, min(60.0, v))
            except Exception:
                pass
            return float(default_tax_pct or 0.0)

        merged["Tax %"] = merged.apply(_tax_pct, axis=1)
        merged["Net"]   = (merged["Gross"] * (1.0 - merged["Tax %"] / 100.0)).round(2)

        # Totals by year (NET)
        dps_by_year = (
            merged.groupby("Year", as_index=False)["Net"].sum()
                  .rename(columns={"Net": "Total Dividend (Net RM)"})
                  .sort_values("Year")
                  .reset_index(drop=True)
        )

        # Per-stock breakdown (both NET & GROSS)
        breakdown = (
            merged.groupby(["Year", "Name"], as_index=False)
                  .agg(**{
                      "Dividend (Net RM)":   ("Net", "sum"),
                      "Dividend (Gross RM)": ("Gross", "sum"),
                      "Held %":              ("Held %", "mean"),
                      "Tax %":               ("Tax %", "mean"),
                  })
                  .sort_values(["Year", "Name"])
                  .reset_index(drop=True)
        )

# KPIs — latest calendar year NET dividend
latest_val = "—"
if not dps_by_year.empty:
    now_year = int(pd.Timestamp.now().year)
    row = dps_by_year[dps_by_year["Year"] == now_year]
    if row.empty:
        row = dps_by_year.tail(1)
    if not row.empty:
        latest_val = f'{float(row["Total Dividend (Net RM)"].iloc[0]):,.2f}'

render_stat_cards(
    [
        {"label": "Projected Dividend (latest year, NET)", "value": latest_val, "badge": "RM", "tone": "good"},
        {"label": "Long Positions", "value": f"{len(filtered):,}", "badge": "Count"},
    ],
    columns=2,
)

# Tables
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(
        dps_by_year,
        use_container_width=True,
        height=min(260, 72 + 28*max(1, min(len(dps_by_year), 8))),
    )
with c2:
    st.dataframe(
        breakdown,
        use_container_width=True,
        height=min(260, 72 + 28*max(1, min(len(breakdown), 8))),
    )

st.caption(
    "Notes: Uses **DPS × Shares × Held-fraction** per year, based on your **OpenDate** and current date. "
    "Tax is **per-stock `TaxPct`** if available on your Ongoing/Open list; otherwise the page-level default. "
    "Proration is evenly spread across the year (doesn’t model actual ex-dates)."
)
