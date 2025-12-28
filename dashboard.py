import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Tariff Impact Severity (Quarter Comparison)", layout="wide")
DATA_PATH = "data/statcan_impact_panel.parquet"
st.caption(
    "Note: Q2 responses largely precede clear tariff direction specification. "
    "Directional impacts (imports vs exports) emerge in Q3–Q4 following tariff implementation."
)


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    for c in ["Period","Quarter","REF_DATE","GEO","Business characteristics","Perspective","Impact_level","UOM"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["Impact_weight"] = pd.to_numeric(df["Impact_weight"], errors="coerce")
    return df.dropna(subset=["VALUE"])

def severity_index(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    x = df[df["Impact_weight"] >= 0].copy()
    if x.empty:
        return pd.DataFrame(columns=group_cols + ["severity_index"])

    x["w_value"] = x["Impact_weight"] * x["VALUE"]
    g = x.groupby(group_cols, as_index=False).agg(
        weighted_sum=("w_value","sum"),
        value_sum=("VALUE","sum")
    )
    g["severity_index"] = g["weighted_sum"] / g["value_sum"].replace({0: pd.NA})
    return g.drop(columns=["weighted_sum","value_sum"])

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

st.title("Tariff Impact Severity (StatCan) — Quarter Comparison")
st.caption("Impact distribution + severity index (0=no impact … 3=high impact) + user-selected quarter deltas.")

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
st.subheader("Impact distribution (mean VALUE) by Quarter")
dist = f.groupby(["Quarter","Perspective","Impact_level"], as_index=False)["VALUE"].mean()

IMPACT_LABEL_MAP = {
    "Level of impact, no impact": "No impact",
    "Level of impact, low impact": "Low",
    "Level of impact, medium impact": "Medium",
    "Level of impact, high impact": "High",
    "Level of impact, unknown": "Unknown",
    "Level of impact, major negative impact": "Major −",
    "Level of impact, minor negative impact": "Minor −",
    "Level of impact, minor positive impact": "Minor +",
    "Level of impact, major positive impact": "Major +",
}

dist["Impact"] = (
    dist["Impact_level"]
    .map(IMPACT_LABEL_MAP)
    .fillna(dist["Impact_level"])
)

fig = px.bar(
    dist,
    x="Impact",
    y="VALUE",
    color="Quarter",
    barmode="group",
    facet_col="Perspective"
)
fig.update_layout(
    height=450,
    margin=dict(l=40, r=40, t=60, b=120)
)
fig.update_xaxes(tickangle=-30)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Severity index by quarter ---
st.subheader("Severity index by Quarter (0–3)")
idx_q = severity_index(f, ["Quarter","Perspective"])
fig = px.bar(idx_q, x="Perspective", y="severity_index", color="Quarter", barmode="group")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Delta severity index by business characteristic (USER SELECTS) ---
st.subheader("Q4 − Q3 delta: severity index by business characteristic")
st.caption(
    "Directional comparison shown only for Q3–Q4, when tariff direction was explicitly reported."
)

idx_bc = severity_index(f, ["Quarter","Perspective","Business characteristics"])
d = delta_between_quarters(
    idx_bc,
    group_cols=["Perspective","Business characteristics"],
    q_from=q_from,
    q_to=q_to,
    require_both=require_both
).dropna(subset=["delta"])

if hide_zeros:
    d = d[d["delta"].abs() > 0.001]

top_n = st.slider("Top N movers", 10, 60, 25)

top_inc = d.sort_values("delta", ascending=False).head(top_n)
top_dec = d.sort_values("delta", ascending=True).head(top_n)

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Largest increases ({q_to} − {q_from})**")

    fig = px.bar(
        top_inc,
        x="delta",
        y="Business characteristics",
        color="Perspective",
        orientation="h"
    )

    fig.update_layout(
        height=max(450, 22 * len(top_inc)),
        margin=dict(l=260, r=40, t=40, b=80),  # ← key fix
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(automargin=False)
    )

    st.plotly_chart(fig, use_container_width=True)


with c2:
    st.markdown(f"**Largest decreases ({q_to} − {q_from})**")

    fig = px.bar(
        top_dec,
        x="delta",
        y="Business characteristics",
        color="Perspective",
        orientation="h"
    )

    fig.update_layout(
        height=max(450, 22 * len(top_dec)),
        margin=dict(l=260, r=40, t=40, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(automargin=False)
    )

    st.plotly_chart(fig, use_container_width=True)
st.divider()

# Optional: show the delta table
with st.expander("Show delta table"):
    st.dataframe(
        d.sort_values(["Perspective", "delta"], ascending=[True, False]),
        use_container_width=True
    )

st.subheader("Data table (filtered)")
st.dataframe(
    f[["Period","Quarter","Perspective","Business characteristics","Impact_level","Impact_weight","VALUE","UOM"]]
    .sort_values(["Quarter","Perspective","Business characteristics","Impact_level"]),
    use_container_width=True
)
