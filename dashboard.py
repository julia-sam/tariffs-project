import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.express as px
import textwrap
import re

st.set_page_config(page_title="Tariff Impact Severity (Quarter Comparison)", layout="wide")
DATA_PATH = "data/statcan_impact_panel.parquet"

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    for c in ["Period","Quarter","REF_DATE","GEO","Business characteristics","Perspective","Impact_level","UOM"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["Impact_weight"] = pd.to_numeric(df["Impact_weight"], errors="coerce")
    return df.dropna(subset=["VALUE"])

def severity_index(df: pd.DataFrame, group_cols: list[str], include_direction: bool = False) -> pd.DataFrame:
    x = df[df["Impact_weight"] >= 0].copy()
    if x.empty:
        return pd.DataFrame(columns=group_cols + (["severity_index","signed_severity_index"] if include_direction else ["severity_index"]))

    x["w_value"] = x["Impact_weight"] * x["VALUE"]

    if include_direction:
        def sign_of(s):
            s = str(s).lower()
            if "negative" in s:
                return -1
            if "positive" in s:
                return 1
            return 0
        x["sign"] = x["Impact_level"].astype(str).apply(sign_of)
        x["w_value_signed"] = x["Impact_weight"] * x["sign"] * x["VALUE"]

    g = x.groupby(group_cols, as_index=False).agg(
        weighted_sum=("w_value", "sum"),
        value_sum=("VALUE", "sum")
    )
    g["severity_index"] = g["weighted_sum"] / g["value_sum"].replace({0: pd.NA})

    if include_direction:
        g_signed = x.groupby(group_cols, as_index=False).agg(weighted_sum_signed=("w_value_signed", "sum"))
        g = g.merge(g_signed, on=group_cols)
        g["signed_severity_index"] = g["weighted_sum_signed"] / g["value_sum"].replace({0: pd.NA})
        return g.drop(columns=["weighted_sum", "value_sum", "weighted_sum_signed"])
    return g.drop(columns=["weighted_sum", "value_sum"])

def quarter_sort_key(q: str) -> int:
    """Sort Q1..Q4 safely."""
    q = str(q).strip().upper()
    return {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.get(q, 99)

def delta_between_quarters(
    idx: pd.DataFrame,
    group_cols: list[str],
    q_from: str,
    q_to: str,
    require_both: bool = True
) -> pd.DataFrame:
    """
    Compute delta = severity_index(q_to) - severity_index(q_from)
    idx is a dataframe like: [Quarter, ...group_cols..., severity_index]
    """
    piv = idx.pivot_table(
        index=group_cols,
        columns="Quarter",
        values="severity_index",
        aggfunc="mean"
    ).reset_index()

    # Ensure columns exist
    if q_from not in piv.columns:
        piv[q_from] = pd.NA
    if q_to not in piv.columns:
        piv[q_to] = pd.NA

    if require_both:
        piv = piv[piv[q_from].notna() & piv[q_to].notna()].copy()

    piv["delta"] = piv[q_to] - piv[q_from]
    piv.rename(columns={q_from: "from_value", q_to: "to_value"}, inplace=True)
    return piv

# st.title("Tariff Impact Severity — Quarter Comparison")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

persp_vals = sorted(df["Perspective"].unique())
persp_sel = st.sidebar.multiselect("Perspective", persp_vals, default=persp_vals)

geo_vals = sorted(df["GEO"].unique())
geo_default = ["Canada"] if "Canada" in geo_vals else geo_vals[:1]
geo_sel = st.sidebar.multiselect("GEO", geo_vals, default=geo_default)

uom_vals = sorted(df["UOM"].unique()) if "UOM" in df.columns else []
uom_sel = st.sidebar.multiselect("UOM", uom_vals, default=uom_vals[:1] if uom_vals else [])

family = st.sidebar.selectbox("Business characteristic family", ["Industry","Employees","Age","All"], index=0)

impact_vals = sorted(df["Impact_level"].unique())
impact_sel = st.sidebar.multiselect("Impact levels", impact_vals, default=impact_vals)

# NEW: quarter selection for delta
quarters = sorted(df["Quarter"].dropna().unique().tolist(), key=quarter_sort_key)
if not quarters:
    st.error("No Quarter values found in the data.")
    st.stop()

st.sidebar.header("Delta (quarter comparison)")
default_to = quarters[-1]
default_from = quarters[-2] if len(quarters) >= 2 else quarters[-1]

q_from = st.sidebar.selectbox("Delta from", quarters, index=quarters.index(default_from))
q_to = st.sidebar.selectbox("Delta to", quarters, index=quarters.index(default_to))

# Prevent user selecting same quarter (allowed but useless)
if q_from == q_to:
    st.sidebar.warning("Delta from and Delta to are the same. Delta will be 0.")

require_both = st.sidebar.checkbox("Require both quarters present", value=True)
hide_zeros = st.sidebar.checkbox("Hide ~zero deltas", value=True)

# Apply filters
f = df[df["Perspective"].isin(persp_sel)].copy()
if "Q2" in f["Quarter"].unique():
    st.info(
        "Q2 data appears primarily under 'Other / unspecified' because tariff direction "
        "was not yet fully identified by respondents at the time of the survey."
    )
f = f[f["GEO"].isin(geo_sel)]
if uom_sel:
    f = f[f["UOM"].isin(uom_sel)]
f = f[f["Impact_level"].isin(impact_sel)]

bc = f["Business characteristics"].astype(str)
if family == "Employees":
    f = f[bc.str.contains("employee", case=False, na=False)]
elif family == "Age":
    f = f[bc.str.contains("Age of business", case=False, na=False)]
elif family == "Industry":
    f = f[bc.str.contains(r"\[\d+\]", regex=True, na=False)]

if f.empty:
    st.warning("No rows match the filters.")
    st.stop()


# --- Distribution by quarter ---
st.subheader("Impact distribution by Quarter")
dist = f.groupby(["Quarter","Perspective","Impact_level"], as_index=False)["VALUE"].mean()

IMPACT_LABEL_MAP = {
    "Level of impact, no impact": "No impact",
    "Level of impact, low impact": "Low",
    "Level of impact, medium impact": "Medium",
    "Level of impact, high impact": "High",
    "Level of impact, unknown": "Unknown",
    "Level of impact, major negative impact": "Major Negative Impact",
    "Level of impact, minor negative impact": "Minor Negative Impact",
    "Level of impact, minor positive impact": "Minor Positive Impact",
    "Level of impact, major positive impact": "Major Positive Impact",
}

IMPACT_ORDER = ["Major Negative Impact", "Minor Negative Impact", "No impact", "Minor Positive Impact", "Major Positive Impact"]

dist["Impact"] = (
    dist["Impact_level"]
    .map(IMPACT_LABEL_MAP)
    .fillna(dist["Impact_level"])
)

dist["Impact"] = pd.Categorical(dist["Impact"], categories=IMPACT_ORDER, ordered=True)

fig = px.bar(
    dist,
    x="Impact",
    y="VALUE",
    color="Quarter",
    barmode="group",
    facet_col="Perspective",
    category_orders={"Impact": IMPACT_ORDER}
)
fig.update_layout(
    height=450,
    margin=dict(l=40, r=40, t=60, b=120)
)
fig.update_xaxes(tickangle=-30)

fig.update_yaxes(title_text="Share of Businesses (%)")
fig.update_yaxes(title_text="", col=2)
fig.update_xaxes(title_text="Perspective: Canadian tariffs on goods purchased (imports)", col=1)
fig.update_xaxes(title_text="Perspective: U.S. tariffs on goods sold (exports)", col=2)


fig.for_each_annotation(
    lambda a: a.update(text=a.text.replace("Perspective=Canadian tariffs on goods purchased (imports)", ""))
)
fig.for_each_annotation(
    lambda a: a.update(text=a.text.replace("Perspective=U.S. tariffs on goods sold (exports)", ""))
)

st.plotly_chart(fig, use_container_width=True)
st.divider()

# --- Change since last quarter (industry level) ---

idx = severity_index(
    f,
    group_cols=["Quarter", "Perspective", "Business characteristics"],
    include_direction=False
)

delta_df = delta_between_quarters(
    idx,
    group_cols=["Perspective", "Business characteristics"],
    q_from=q_from,
    q_to=q_to,
    require_both=require_both
)

if hide_zeros:
    delta_df = delta_df[delta_df["delta"].abs() > 0.01]

if delta_df.empty:
    st.warning("No industry-level changes available for the selected quarters.")
    st.stop()

# ensure Industry column exists (used later for labels)
if "Industry" not in delta_df.columns and "Business characteristics" in delta_df.columns:
    delta_df["Industry"] = delta_df["Business characteristics"]

# Sort by magnitude of change
delta_df = delta_df.sort_values("delta", ascending=False)

st.subheader("Industry-level change in trade pressure")

# Clean industry labels
def short_label(text):
    return re.sub(r"\s*\[\d+\]$", "", str(text))

delta_df["Industry_short"] = delta_df["Industry"].apply(short_label)

# Direction label
delta_df["Direction"] = delta_df["delta"].apply(
    lambda x: "More pressure" if x > 0 else "Relief"
)

# Sort industries within each perspective by absolute change
delta_df["abs_delta"] = delta_df["delta"].abs()
delta_df = delta_df.sort_values(
    ["Perspective", "abs_delta"],
    ascending=[True, False]
)

# Make Industry a categorical to preserve sorting — use unique categories only
unique_cats = list(pd.unique(delta_df["Industry_short"]))
delta_df["Industry_short"] = pd.Categorical(
    delta_df["Industry_short"],
    categories=unique_cats,
    ordered=True
)

fig = px.bar(
    delta_df,
    x="delta",
    y="Industry_short",
    color="Direction",
    facet_col="Perspective",
    orientation="h",
    color_discrete_map={
        "More pressure": "#dc2626",
        "Relief": "#16a34a",
    },
    hover_data={
        "Industry": True,
        "delta": ":.3f",
        "Industry_short": False,
        "abs_delta": False,
    }
)

# Zero line (critical for interpretation)
fig.add_vline(
    x=0,
    line_width=1,
    line_color="black",
    opacity=0.6
)

# Axis + layout cleanup
fig.update_layout(
    height=700,
    showlegend=True,
    margin=dict(l=160, r=40, t=60, b=40),
)

fig.update_xaxes(
    title="Change in severity index (Δ quarter-to-quarter)",
    zeroline=False,
    showgrid=True,
)

fig.update_yaxes(
    title="",
    autorange="reversed",  # biggest change at top
)

# Clean facet titles
fig.for_each_annotation(
    lambda a: a.update(
        text=a.text
        .replace("Perspective=", "")
        .replace("Canadian tariffs on goods purchased (imports)", "Imports")
        .replace("U.S. tariffs on goods sold (exports)", "Exports")
    )
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Data table (filtered)")
st.dataframe(
    f[["Period","Quarter","Perspective","Business characteristics","Impact_level","Impact_weight","VALUE","UOM"]]
    .rename(columns={"VALUE": "Number of Businesses"})
    .sort_values(["Quarter","Perspective","Business characteristics","Impact_level"]),
    use_container_width=True
)
