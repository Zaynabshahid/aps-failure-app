import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(
    page_title="APS Failure Diagnosis | Zaynab Shahid",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

COST_FP = 10
COST_FN = 500

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif; }

.hero {
    padding: 2.4rem 2.2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: #f4f1ea;
    margin-bottom: 1.6rem;
}
.hero h1 { color: #f4f1ea; font-size: 2.4rem; margin-bottom: 0.2rem; }
.hero p { color: #cfd8dc; font-size: 1.05rem; line-height: 1.6; }
.hero .byline { color: #e8c07d; font-weight: 500; letter-spacing: 0.03em; margin-top: 0.6rem; }

.chapter-tag {
    display: inline-block;
    background: #e8c07d;
    color: #0f2027;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.insight-box {
    background: #f7f5f0;
    border-left: 4px solid #2c5364;
    padding: 1rem 1.3rem;
    border-radius: 6px;
    margin: 0.9rem 0 1.4rem 0;
    font-size: 0.98rem;
    line-height: 1.65;
    color: #1c1c1c;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e5e1d8;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.footer-note {
    text-align: center;
    color: #8a8a8a;
    font-size: 0.82rem;
    padding: 1.6rem 0 0.8rem 0;
    border-top: 1px solid #eee;
    margin-top: 2.2rem;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Reading the fleet records...")
def load_data():
    train = pd.read_csv("Dataset/aps_failure_training_set.csv", skiprows=20, na_values="na")
    test = pd.read_csv("Dataset/aps_failure_test_set.csv", skiprows=20, na_values="na")
    train["class"] = (train["class"] == "pos").astype(int)
    test["class"] = (test["class"] == "pos").astype(int)
    return train, test


try:
    train_df, test_df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="chapter-tag">A Data Story</div>
    <h1>The Truck That Would Not Say Where It Hurt</h1>
    <p>Every truck in this dataset has already failed. None of them are healthy, and none of them
    will tell a technician, in plain words, which system gave out first. This is the story of how
    170 anonymized sensors, some of them silent, some of them loud, were made to answer a question
    that used to cost a service center an afternoon of guesswork: is this an Air Pressure System
    failure, or something else.</p>
    <p class="byline">Written and built by Zaynab Shahid</p>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error(
        "The dataset files 'aps_failure_training_set.csv' and 'aps_failure_test_set.csv' were not "
        "found in the app folder. Place both files in the same directory as this script and rerun."
    )
    st.stop()


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.markdown("### Chapters")
chapter = st.sidebar.radio(
    label="Navigate the story",
    options=[
        "I. The Fleet",
        "II. The Silence in the Sensors",
        "III. The Anonymous Families",
        "IV. Who Really Separates the Two",
        "V. A Different Magnitude",
        "VI. The Cost of Guessing",
        "VII. Try the Threshold Yourself",
        "VIII. Closing Notes",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this app**\n\n"
    "Built for the Datadive GDGoC datathon. Explores the APS Failure at Scania Trucks dataset "
    "through the same investigative questions used in the accompanying notebook, presented here "
    "as an interactive narrative rather than a static report."
)
st.sidebar.markdown("---")
st.sidebar.caption("Created by Zaynab Shahid")


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def total_cost(y_true, y_pred, cost_fp=COST_FP, cost_fn=COST_FN):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return cost_fp * fp + cost_fn * fn, tn, fp, fn, tp


@st.cache_data(show_spinner=False)
def get_missing_stats(df):
    miss_pct = df.drop(columns="class").isna().mean().sort_values(ascending=False) * 100
    return miss_pct


@st.cache_data(show_spinner=False)
def get_point_biserial(df):
    y = df["class"]
    out = {}
    for col in df.columns:
        if col == "class":
            continue
        valid = df[col].notna()
        if valid.sum() < 100:
            continue
        r, _ = stats.pointbiserialr(y[valid], df.loc[valid, col])
        out[col] = r
    return pd.Series(out).dropna()


# ========================================================================
# CHAPTER I
# ========================================================================
if chapter == "I. The Fleet":
    st.markdown('<div class="chapter-tag">Chapter I</div>', unsafe_allow_html=True)
    st.header("The Fleet")

    st.markdown(
        "Sixty thousand trucks arrive in the training records. Every one of them has already "
        "broken down. The service center does not ask whether something is wrong. It asks what."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Trucks in training set", f"{len(train_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Trucks in test set", f"{len(test_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Anonymized sensors", "170")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("APS share, training", f"{train_df['class'].mean()*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    dist_choice = st.radio("View class balance for:", ["Training set", "Test set"], horizontal=True)
    plot_df = train_df if dist_choice == "Training set" else test_df
    counts = plot_df["class"].value_counts().rename({0: "Other failure", 1: "APS failure"})

    fig = px.bar(
        x=counts.index, y=counts.values,
        color=counts.index,
        color_discrete_map={"Other failure": "#2c5364", "APS failure": "#e8c07d"},
        labels={"x": "Root cause", "y": "Number of trucks"},
        text=counts.values,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, title=f"Class balance, {dist_choice.lower()}", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    Only a small fraction of these failures trace back to the Air Pressure System. That rarity is
    the whole shape of the problem. A model that predicts every truck as a non-APS failure would
    be right almost every time and useful never, since it would miss every genuine APS case that
    walked through the gate.
    </div>
    """, unsafe_allow_html=True)


# ========================================================================
# CHAPTER II
# ========================================================================
elif chapter == "II. The Silence in the Sensors":
    st.markdown('<div class="chapter-tag">Chapter II</div>', unsafe_allow_html=True)
    st.header("The Silence in the Sensors")

    st.markdown(
        "Not every sensor speaks on every truck. Some readings are missing so often that silence "
        "becomes a pattern of its own, and patterns, in this dataset, are rarely accidental."
    )

    miss_pct = get_missing_stats(train_df)
    n_top = st.slider("Number of sensors to display", 5, 30, 15)

    fig = px.bar(
        x=miss_pct.head(n_top).values[::-1],
        y=miss_pct.head(n_top).index[::-1],
        orientation="h",
        color=miss_pct.head(n_top).values[::-1],
        color_continuous_scale="Teal",
        labels={"x": "Percent missing", "y": "Sensor"},
    )
    fig.update_layout(title="Sensors that go quiet most often", height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    missing_ind = train_df.drop(columns="class").isna().astype(int)
    missing_ind["class"] = train_df["class"].values
    corr_missing = missing_ind.corr()["class"].drop("class").sort_values(key=abs, ascending=False)

    fig2 = px.bar(
        x=corr_missing.head(12).values[::-1],
        y=corr_missing.head(12).index[::-1],
        orientation="h",
        color=corr_missing.head(12).values[::-1],
        color_continuous_scale="RdBu",
        labels={"x": "Correlation with APS failure", "y": "Sensor"},
    )
    fig2.update_layout(title="Where silence itself becomes a clue", height=440, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    Certain sensors are missing at different rates depending on which system actually failed.
    That is not noise to be filled in and forgotten. It is treated in the underlying model as a
    feature in its own right, since a missing reading here is often the truck saying something
    before it says anything else.
    </div>
    """, unsafe_allow_html=True)


# ========================================================================
# CHAPTER III
# ========================================================================
elif chapter == "III. The Anonymous Families":
    st.markdown('<div class="chapter-tag">Chapter III</div>', unsafe_allow_html=True)
    st.header("The Anonymous Families")

    st.markdown(
        "The sensors carry no names, only codes. But codes that share a prefix behave like "
        "relatives, rising and falling together. Ten of these families look like histogram bins, "
        "counting how often a signal fell into one bucket or another."
    )

    import re
    from collections import defaultdict

    base_groups = defaultdict(list)
    for col in train_df.columns:
        if col == "class":
            continue
        m = re.match(r"^([a-z]{2})_(\d{3})$", col)
        if m:
            base_groups[m.group(1)].append(col)
    hist_groups = {k: sorted(v) for k, v in base_groups.items() if len(v) >= 5}

    family_choice = st.selectbox("Choose a sensor family to examine", sorted(hist_groups.keys()))
    bins = hist_groups[family_choice]
    bin_means = train_df.groupby("class")[bins].mean()
    bin_share = bin_means.div(bin_means.sum(axis=1), axis=0)

    plot_data = bin_share.T.reset_index().melt(id_vars="index", var_name="class", value_name="share")
    plot_data["class"] = plot_data["class"].map({0: "Other failure", 1: "APS failure"})
    plot_data["bin"] = plot_data["index"].str.replace(f"{family_choice}_", "bin ", regex=False)

    fig = px.bar(
        plot_data, x="bin", y="share", color="class", barmode="group",
        color_discrete_map={"Other failure": "#2c5364", "APS failure": "#e8c07d"},
        labels={"share": "Share of total family mass", "bin": "Bin"},
    )
    fig.update_layout(title=f'Family "{family_choice}": where the mass sits, by class', height=460)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    Bins from the same family rarely move independently. When one bin swells, its neighbors
    usually shift as well, which is why these families are engineered as group level summaries
    rather than ten unrelated columns competing for the model's attention.
    </div>
    """, unsafe_allow_html=True)


# ========================================================================
# CHAPTER IV
# ========================================================================
elif chapter == "IV. Who Really Separates the Two":
    st.markdown('<div class="chapter-tag">Chapter IV</div>', unsafe_allow_html=True)
    st.header("Who Really Separates the Two")

    st.markdown(
        "Out of 170 sensors, only a handful carry most of the weight. This chapter names them."
    )

    pb_corr = get_point_biserial(train_df)
    pb_sorted = pb_corr.reindex(pb_corr.abs().sort_values(ascending=False).index)
    n_show = st.slider("Number of top sensors to reveal", 5, 25, 15, key="pb_slider")

    top_n = pb_sorted.head(n_show)
    fig = px.bar(
        x=top_n.values[::-1], y=top_n.index[::-1], orientation="h",
        color=top_n.values[::-1], color_continuous_scale="RdBu",
        labels={"x": "Correlation with APS failure", "y": "Sensor"},
    )
    fig.update_layout(title="Individually strongest predictors of APS failure", height=520, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    A short list of sensors carries a disproportionate share of the individually measurable
    signal in this dataset. This does not make the remaining sensors worthless, since the final
    model can still exploit weaker interactions between them, but it does mean the strongest
    story in this data is told by relatively few voices.
    </div>
    """, unsafe_allow_html=True)


# ========================================================================
# CHAPTER V
# ========================================================================
elif chapter == "V. A Different Magnitude":
    st.markdown('<div class="chapter-tag">Chapter V</div>', unsafe_allow_html=True)
    st.header("A Different Magnitude")

    st.markdown(
        "Statistical significance says two groups differ. It does not say by how much, or in what "
        "way. This chapter looks at the shape of that difference directly."
    )

    pb_corr = get_point_biserial(train_df)
    pb_sorted = pb_corr.reindex(pb_corr.abs().sort_values(ascending=False).index)
    sensor_choice = st.selectbox("Choose a sensor to inspect", pb_sorted.head(20).index.tolist())

    plot_data = train_df[[sensor_choice, "class"]].dropna().copy()
    plot_data["log_value"] = np.log1p(plot_data[sensor_choice].clip(lower=0))
    plot_data["Root cause"] = plot_data["class"].map({0: "Other failure", 1: "APS failure"})

    fig = px.histogram(
        plot_data, x="log_value", color="Root cause", barmode="overlay", opacity=0.6,
        histnorm="probability density",
        color_discrete_map={"Other failure": "#2c5364", "APS failure": "#e8c07d"},
        labels={"log_value": f"log(1 + {sensor_choice})"},
    )
    fig.update_layout(title=f"Distribution of {sensor_choice} by root cause", height=460)
    st.plotly_chart(fig, use_container_width=True)

    median_neg = plot_data.loc[plot_data["class"] == 0, sensor_choice].median()
    median_pos = plot_data.loc[plot_data["class"] == 1, sensor_choice].median()
    c1, c2 = st.columns(2)
    c1.metric(f"Median, other failure", f"{median_neg:,.0f}")
    c2.metric(f"Median, APS failure", f"{median_pos:,.0f}")

    st.markdown("""
    <div class="insight-box">
    On several of the strongest sensors, APS trucks do not merely nudge higher or lower, they
    often sit in a distinctly different operating range. That kind of separation favors models
    that can draw sharp, nonlinear boundaries rather than a single straight cutoff.
    </div>
    """, unsafe_allow_html=True)


# ========================================================================
# CHAPTER VI
# ========================================================================
elif chapter == "VI. The Cost of Guessing":
    st.markdown('<div class="chapter-tag">Chapter VI</div>', unsafe_allow_html=True)
    st.header("The Cost of Guessing")

    st.markdown(
        "A missed APS failure costs fifty times what a false alarm costs. That single fact "
        "reshapes every decision made after the exploratory work is done."
    )

    c1, c2 = st.columns(2)
    c1.metric("Cost of a false alarm", f"${COST_FP}")
    c2.metric("Cost of a missed APS failure", f"${COST_FN}")

    fn_counts = list(range(0, 21))
    costs_if_missed = [f * COST_FN for f in fn_counts]
    fig = px.line(
        x=fn_counts, y=costs_if_missed, markers=True,
        labels={"x": "Number of APS failures missed", "y": "Cost incurred (dollars)"},
    )
    fig.update_traces(line_color="#e8c07d")
    fig.update_layout(title="How quickly missed diagnoses become expensive", height=420)
    st.plotly_chart(fig, use_container_width=True)

    majority_preds = np.zeros(len(train_df))
    maj_cost, tn, fp, fn, tp = total_cost(train_df["class"].values, majority_preds)

    st.markdown(f"""
    <div class="insight-box">
    A policy that never flags a single truck for APS inspection would still be right about
    98 percent of the time on this training set, and it would still cost {maj_cost:,} dollars,
    entirely from missed diagnoses. That number is the floor every model in this project is
    measured against, not accuracy.
    </div>
    """, unsafe_allow_html=True)


# ========================================================================
# CHAPTER VII
# ========================================================================
elif chapter == "VII. Try the Threshold Yourself":
    st.markdown('<div class="chapter-tag">Chapter VII</div>', unsafe_allow_html=True)
    st.header("Try the Threshold Yourself")

    st.markdown(
        "The single strongest sensor in this dataset, on its own, already beats the do nothing "
        "policy. Move the threshold below and watch the cost respond in real time, the same "
        "exercise the underlying model performs at a larger scale before a single prediction "
        "is trusted."
    )

    pb_corr = get_point_biserial(train_df)
    pb_sorted = pb_corr.reindex(pb_corr.abs().sort_values(ascending=False).index)
    lead_sensor = pb_sorted.index[0]

    vals = train_df[lead_sensor].fillna(train_df[lead_sensor].median()).values
    med_pos = train_df.loc[train_df["class"] == 1, lead_sensor].median()
    med_neg = train_df.loc[train_df["class"] == 0, lead_sensor].median()
    direction = 1 if med_pos >= med_neg else -1
    score = direction * vals
    ranks = stats.rankdata(score) / len(score)
    y_true = train_df["class"].values

    threshold = st.slider(
        f"Decision threshold on ranked score of {lead_sensor}",
        min_value=0.0, max_value=1.0, value=0.5, step=0.005
    )
    preds = (ranks >= threshold).astype(int)
    cost, tn, fp, fn, tp = total_cost(y_true, preds)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("False positives", f"{fp:,}")
    c2.metric("False negatives", f"{fn:,}")
    c3.metric("Total cost", f"${cost:,}")
    c4.metric("Trucks flagged", f"{fp + tp:,}")

    thresholds = np.linspace(0.01, 0.99, 150)
    sweep_costs = [total_cost(y_true, (ranks >= t).astype(int))[0] for t in thresholds]
    best_t = thresholds[int(np.argmin(sweep_costs))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=sweep_costs, mode="lines", line=dict(color="#2c5364"), name="Cost curve"))
    fig.add_vline(x=threshold, line_dash="dot", line_color="#e8c07d", annotation_text="Your threshold")
    fig.add_vline(x=best_t, line_dash="dash", line_color="#c0392b", annotation_text="Cost minimizing point")
    fig.update_layout(
        title=f"Cost across thresholds, single sensor rule on {lead_sensor}",
        xaxis_title="Threshold", yaxis_title="Total training cost", height=460
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    The cost minimizing threshold for this single sensor sits near {best_t:.3f}, well below the
    naive midpoint of 0.5. That leftward pull is not a quirk of this one sensor. It shows up
    again in the full model, for the same underlying reason: missing a real APS failure costs
    far more than a false alarm, so the sensible cutoff sits lower than instinct would suggest.
    </div>
    """, unsafe_allow_html=True)


# ========================================================================
# CHAPTER VIII
# ========================================================================
elif chapter == "VIII. Closing Notes":
    st.markdown('<div class="chapter-tag">Chapter VIII</div>', unsafe_allow_html=True)
    st.header("Closing Notes")

    st.markdown(
        """
        A truck cannot describe its own failure. It can only offer readings, some present, some
        missing, some sitting quietly in a range no healthy truck would occupy. The work in this
        project was to listen to all of that carefully enough to tell two kinds of failure apart,
        and to do it in a way that respects what a wrong answer actually costs a service center,
        not just how often the answer happens to be right.

        The chapters above are not a substitute for the full notebook, where every one of these
        findings is tested formally and carried through to a final model and a justified decision
        threshold. This app exists to make those findings walkable, so that the reasoning behind
        the model is something anyone can explore, not just something buried in a script.
        """
    )

    st.markdown("---")
    st.markdown(
        "**Project:** APS Failure Diagnosis, Scania Trucks  \n"
        "**Author:** Zaynab Shahid  \n"
        "**Built for:** Datadive GDGoC '26"
    )

st.markdown(
    '<div class="footer-note">Built by Zaynab Shahid for the Datadive GDGoC \'26 datathon</div>',
    unsafe_allow_html=True
)