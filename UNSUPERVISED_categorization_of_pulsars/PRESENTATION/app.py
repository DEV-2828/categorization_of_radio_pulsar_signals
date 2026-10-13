import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_loader import (
    load_data,
    FEATURE_NAMES,
    FEATURE_LABELS,
    CLUSTER_METADATA,
    PAPER_TARGETS,
    get_base_paths
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pulsar Categorization | Unsupervised ML Defense",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Astrophysics Observatory Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Gradient hero title */
    .hero-title {
        background: linear-gradient(135deg, #00d4ff 0%, #7928ca 50%, #ff007f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    /* Stat cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }
    .stat-val { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
    .stat-lbl { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; }

    /* Cluster pill badges */
    .pill-c1 {
        background: rgba(0,212,255,0.15); color: #00d4ff;
        padding: 4px 10px; border-radius: 6px; font-weight: 600;
        font-size: 0.85rem; border: 1px solid rgba(0,212,255,0.3);
    }
    .pill-c2 {
        background: rgba(255,159,28,0.15); color: #ff9f1c;
        padding: 4px 10px; border-radius: 6px; font-weight: 600;
        font-size: 0.85rem; border: 1px solid rgba(255,159,28,0.3);
    }
    .pill-c3 {
        background: rgba(255,0,127,0.15); color: #ff007f;
        padding: 4px 10px; border-radius: 6px; font-weight: 600;
        font-size: 0.85rem; border: 1px solid rgba(255,0,127,0.3);
    }

    /* Callout box */
    .astronomy-callout {
        background: linear-gradient(135deg, rgba(15,23,42,0.8) 0%, rgba(30,41,59,0.7) 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    /* Scorecard color rows */
    .score-row-pass {
        background: rgba(34,197,94,0.12);
        border-left: 3px solid #22c55e;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-size: 0.9rem;
        color: #f8fafc;
    }
    .score-row-warn {
        background: rgba(234,179,8,0.12);
        border-left: 3px solid #eab308;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-size: 0.9rem;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_dataset():
    return load_data()

@st.cache_data
def get_paths():
    return get_base_paths()

try:
    df, pca_model = get_dataset()
    paths = get_paths()
except FileNotFoundError as e:
    st.error(f"⚠️ **Data file missing:** {e}")
    st.info("Run `DATA/data_extractor.py`, then `UNSUPERVISED_MODELS/K-means_clustering/kmeans_model.py`, and `UNSUPERVISED_MODELS/SOM/som_model.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 18px 0 10px 0;">
        <span style="font-size:2.8rem;">🌌</span>
        <div style="font-size:1.05rem; font-weight:700; color:#00d4ff; letter-spacing:1px; margin-top:6px;">PULSAR OBSERVATORY</div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:2px;">Unsupervised ML Presentation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛰️ Navigation")

    section = st.radio(
        "Select Section:",
        [
            "1. 🪐 Scope & Problem Definition",
            "2. 🌀 PCA & The Boomerang Manifold",
            "3. 🎯 K-Means Sub-populations (k=3)",
            "4. 🧠 Self-Organizing Map (SOM)",
            "5. ⚡ Live Pulsar Sandbox",
            "6. 🏆 Paper Parity & Final Scorecard"
        ],
        index=0  # ← default to Section 1
    )

    st.markdown("---")
    st.markdown("### 🔭 Quick Reference")
    st.markdown("""
    - **Dataset:** HTRU-2 Survey
    - **Pulsars:** 1,639 positive signals
    - **Features:** 8 unscaled moments
    - **Paper:** *Debesai, Gutierrez, Koyluoglu*
    """)

    st.markdown("---")
    # ── Download buttons ────────────────────────────────────────────────────
    st.markdown("### 📥 Downloads")
    if os.path.exists(paths['report']):
        with open(paths['report'], 'rb') as f:
            st.download_button(
                label="📄 Download Full Report (.md)",
                data=f.read(),
                file_name="UNSUPERVISED_MODELS_REPORT.md",
                mime="text/markdown"
            )
    if os.path.exists(paths['kmeans']):
        with open(paths['kmeans'], 'rb') as f:
            st.download_button(
                label="📊 Download K-Means Clusters (.csv)",
                data=f.read(),
                file_name="kmeans_clustered_pulsars.csv",
                mime="text/csv"
            )
    if os.path.exists(paths['som']):
        with open(paths['som'], 'rb') as f:
            st.download_button(
                label="🧠 Download SOM Clusters (.csv)",
                data=f.read(),
                file_name="som_clustered_pulsars.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.caption("Machine Learning SEM-5 | Unsupervised Module Defense")

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Categorization of Radio Pulsar Signals</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Unsupervised Sub-population Discovery · Manifold Analysis · Stanford CS 229 Parity Benchmark</div>',
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SCOPE & PROBLEM DEFINITION
# ═════════════════════════════════════════════════════════════════════════════
if section.startswith("1."):
    st.subheader("1. Project Scope & The Unsupervised Challenge")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "1,639", "True Pulsars Analyzed"),
        (c2, "8",     "Continuous Features"),
        (c3, "0.4413","K-Means Silhouette Score"),
        (c4, "86.04%","Variance via 2 PCs"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{val}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="astronomy-callout">
        <b>💡 Why Unsupervised Learning?</b><br>
        In radio astronomy surveys, raw candidates are averaged in frequency and folded across phase into
        8 statistical moments — discarding ground-truth rotational period and dispersion distance labels.
        While supervised models answer <i>"Is this pulsar or RFI noise?"</i>,
        <b>unsupervised learning is required to discover the intrinsic physical sub-types within the confirmed pulsars</b>.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("#### The 8 Extracted Statistical Features")
        feat_df = pd.DataFrame([
            {"#": i+1, "Feature": f, "Astrophysical Meaning": FEATURE_LABELS[f]}
            for i, f in enumerate(FEATURE_NAMES)
        ])
        feat_df["Astrophysical Meaning"] = [
            "Average brightness across folded pulse phase",
            "Width / spread of the pulse profile",
            "Sharpness / peakedness of the pulse",
            "Asymmetry between leading & trailing pulse edges",
            "Mean signal-to-noise ratio across Dispersion trials",
            "Dispersion peak prominence and sharpness",
            "Localization of the true dispersion measure",
            "Tail asymmetry in dispersion measure search space"
        ]
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

        st.markdown("#### Feature Distribution Explorer")
        feat_sel = st.selectbox(
            "Inspect feature distribution across clusters:",
            FEATURE_NAMES,
            format_func=lambda f: FEATURE_LABELS[f],
            key="feat_sel_s1"
        )
        fig_hist = px.histogram(
            df, x=feat_sel, color='kmeans_cluster',
            color_discrete_map={'Cluster 1': '#00d4ff', 'Cluster 2': '#ff9f1c', 'Cluster 3': '#ff007f'},
            barmode='overlay', opacity=0.7, nbins=60,
            title=f"Distribution of {FEATURE_LABELS[feat_sel]} by Cluster",
            template="plotly_dark"
        )
        fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown("#### Methodological Decision: Unscaled Features")
        st.info("""
        **Why Raw Features Instead of StandardScaler?**
        - **Replication of Debesai et al.:** The paper trained K-Means and SOM on unscaled features,
          confirmed by centroid values in Table 2 & Table 3.
        - **Physical Signal Dominance:** DM-SNR features (`skewness_dmsnr_curve`, `mean_dmsnr_curve`)
          have naturally larger variance — the primary discriminative signal separating distant,
          canonical, and millisecond pulsar candidates.
        """)

        st.markdown("#### Dataset Sample Preview (First 5 True Pulsars)")
        st.dataframe(df[FEATURE_NAMES].head(5), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PCA & THE BOOMERANG MANIFOLD
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("2."):
    st.subheader("2. Dimensionality Reduction & The 'Boomerang' Manifold")

    st.markdown("""
    Projecting the 8-dimensional unscaled pulsar space onto the first 2 principal components
    explains **86.04%** of total variance — **PC1: 61.06%**, **PC2: 24.98%**.
    """)

    col_ctrl, col_plot = st.columns([0.3, 0.7])

    with col_ctrl:
        st.markdown("#### 🎛️ Manifold Controls")
        color_choice = st.radio(
            "Color Points By:",
            ["K-Means Clusters (k=3)", "SOM Macro-clusters", "Single Color (Geometry only)", "DM-SNR Skewness (Driver)"],
            index=0
        )
        plot_dim = st.radio("Mode:", ["2D Manifold", "3D Manifold"], horizontal=True)
        point_opacity = st.slider("Opacity:", 0.2, 1.0, 0.75, 0.05)
        point_size    = st.slider("Marker Size:", 3, 12, 6)

        st.markdown("---")
        st.markdown("#### 🔬 Geometry Breakdown")
        st.markdown("""
        - **Left arm:** PC1 rises, PC2 falls. Driven by falling `mean_dmsnr_curve`.
        - **Right arm:** Both PC1 and PC2 rise. Driven by exploding `skewness_dmsnr_curve`.
        - **Vertex tip:** Exclusively occupied by the 44 extreme Cluster 3 pulsars.
        """)

    with col_plot:
        COLOR_MAP = {
            'Cluster 1': '#00d4ff', 'Cluster 2': '#ff9f1c', 'Cluster 3': '#ff007f'
        }
        SOM_COLOR_MAP = {
            'SOM Cluster 1': '#00d4ff', 'SOM Cluster 2': '#ff9f1c', 'SOM Cluster 3': '#ff007f'
        }

        hover_data = {
            'PC1': ':.2f', 'PC2': ':.2f',
            'mean_integrated_profile': ':.1f',
            'mean_dmsnr_curve': ':.1f',
            'skewness_dmsnr_curve': ':.1f'
        }

        is_continuous = (color_choice == "DM-SNR Skewness (Driver)")
        color_col = {
            "K-Means Clusters (k=3)": 'kmeans_cluster',
            "SOM Macro-clusters":      'som_cluster',
            "Single Color (Geometry only)": None,
            "DM-SNR Skewness (Driver)": 'skewness_dmsnr_curve'
        }[color_choice]
        color_map = COLOR_MAP if color_choice == "K-Means Clusters (k=3)" else (
                    SOM_COLOR_MAP if color_choice == "SOM Macro-clusters" else None)

        if plot_dim == "2D Manifold":
            common = dict(x='PC1', y='PC2', opacity=point_opacity,
                          hover_data=hover_data, template="plotly_dark")
            if is_continuous:
                fig = px.scatter(df, color='skewness_dmsnr_curve',
                                 color_continuous_scale='Turbo',
                                 title="PCA 'Boomerang' — colored by DM-SNR Skewness", **common)
            elif color_col:
                fig = px.scatter(df, color=color_col, color_discrete_map=color_map,
                                 title=f"PCA 'Boomerang' — {color_choice}", **common)
            else:
                fig = px.scatter(df, color_discrete_sequence=['#38bdf8'],
                                 title="PCA 'Boomerang' — Raw Geometry", **common)
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            common3 = dict(x='PC1', y='PC2', z='mean_dmsnr_curve',
                           opacity=point_opacity, hover_data=hover_data,
                           template="plotly_dark",
                           title="3D Manifold: PC1 × PC2 × Mean DM-SNR")
            if is_continuous:
                fig3 = px.scatter_3d(df, color='skewness_dmsnr_curve',
                                     color_continuous_scale='Turbo', **common3)
            elif color_col:
                fig3 = px.scatter_3d(df, color=color_col, color_discrete_map=color_map, **common3)
            else:
                fig3 = px.scatter_3d(df, color_discrete_sequence=['#38bdf8'], **common3)
            fig3.update_traces(marker=dict(size=point_size))
            fig3.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 PCA Feature Loadings (Eigenvector Weights)")

    comp = pca_model.components_
    loadings_df = pd.DataFrame({
        'Feature': [FEATURE_LABELS[f] for f in FEATURE_NAMES],
        'PC1 Loading (61.06%)': comp[0],
        'PC2 Loading (24.98%)': comp[1]
    })

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        fig_b1 = px.bar(loadings_df, x='PC1 Loading (61.06%)', y='Feature', orientation='h',
                        title="PC1 Weights (Dominated by Skewness DM-SNR)",
                        template="plotly_dark", color='PC1 Loading (61.06%)',
                        color_continuous_scale='Tealgrn')
        fig_b1.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_b1, use_container_width=True)
    with col_l2:
        fig_b2 = px.bar(loadings_df, x='PC2 Loading (24.98%)', y='Feature', orientation='h',
                        title="PC2 Weights (Co-dominated by Skewness & Mean SNR)",
                        template="plotly_dark", color='PC2 Loading (24.98%)',
                        color_continuous_scale='Sunset')
        fig_b2.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_b2, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — K-MEANS SUB-POPULATIONS
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("3."):
    st.subheader("3. K-Means Clustering: Discovery of 3 Pulsar Sub-types")

    # ── Optimal k selection — interactive Plotly charts ───────────────────
    st.markdown("#### Optimal Cluster Count Selection (k = 2 to 7)")

    k_vals       = [2, 3, 4, 5, 6, 7]
    wcss_vals    = [13_419_890, 4_115_339, 2_629_004, 1_894_442, 1_498_386, 1_238_276]
    sil_vals     = [0.3621, 0.4413, 0.3874, 0.3559, 0.3401, 0.3182]

    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        fig_elbow = px.line(
            x=k_vals, y=wcss_vals,
            markers=True,
            labels={'x': 'Number of Clusters (k)', 'y': 'WCSS'},
            title='Elbow Curve — WCSS drops sharply at k=3',
            template="plotly_dark"
        )
        fig_elbow.add_vline(x=3, line_dash="dash", line_color="#ff007f",
                            annotation_text="k=3 (elbow)", annotation_position="top right")
        fig_elbow.update_traces(line_color='#00d4ff', marker_color='#00d4ff')
        fig_elbow.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_diag2:
        fig_sil = px.line(
            x=k_vals, y=sil_vals,
            markers=True,
            labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'},
            title='Silhouette Score — peaks at k=3 (0.4413)',
            template="plotly_dark"
        )
        fig_sil.add_vline(x=3, line_dash="dash", line_color="#ff007f",
                          annotation_text="k=3 (0.4413)", annotation_position="top right")
        fig_sil.update_traces(line_color='#ff9f1c', marker_color='#ff9f1c')
        fig_sil.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")
    st.markdown("#### The 3 Identified Astrophysical Sub-populations")

    col_c1, col_c2, col_c3 = st.columns(3)
    for col, cname, pill_cls, stats_html in [
        (col_c1, 'Cluster 1', 'pill-c1',
         '• Mean IP: <b>69.2</b><br>• Mean DM-SNR: <b>20.6</b><br>• Skewness DM-SNR: <b>18.8</b>'),
        (col_c2, 'Cluster 2', 'pill-c2',
         '• Mean IP: <b>34.4</b> (Dimmer)<br>• Mean DM-SNR: <b>99.1</b> (Massive)<br>• Skewness IP: <b>24.3</b> (Asymmetric)'),
        (col_c3, 'Cluster 3', 'pill-c3',
         '• Mean IP: <b>94.9</b> (Brightest)<br>• Mean DM-SNR: <b>1.3</b> (Narrow)<br>• Skewness DM-SNR: <b>255.6</b> (Extreme Outlier)'),
    ]:
        meta = CLUSTER_METADATA[cname]
        n = int(df[df['kmeans_cluster'] == cname].shape[0]) if 'kmeans_cluster' in df.columns else meta['target_size']
        pct = round(100 * n / len(df), 1)
        with col:
            st.markdown(f"""
            <div class="stat-card" style="border-top: 4px solid {meta['color']};">
                <span class="{pill_cls}">{cname} · {n:,} Pulsars ({pct}%)</span>
                <h4 style="margin-top:10px; color:#f8fafc;">{meta['name']}</h4>
                <p style="font-size:0.9rem; color:#cbd5e1;">{meta['description']}</p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <div style="font-size:0.85rem; color:#94a3b8;">{stats_html}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📡 8D Centroid Comparison — Radar Chart")

    centroids_raw  = df.groupby('kmeans_cluster')[FEATURE_NAMES].mean().loc[['Cluster 1', 'Cluster 2', 'Cluster 3']]
    min_vals       = df[FEATURE_NAMES].min()
    max_vals       = df[FEATURE_NAMES].max()
    centroids_norm = (centroids_raw - min_vals) / (max_vals - min_vals)
    categories     = [FEATURE_LABELS[f] for f in FEATURE_NAMES]

    fig_radar = go.Figure()
    for cname in ['Cluster 1', 'Cluster 2', 'Cluster 3']:
        r_vals = centroids_norm.loc[cname].values.tolist() + [centroids_norm.loc[cname].values[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=categories + [categories[0]],
            fill='toself',
            name=f"{cname} — {CLUSTER_METADATA[cname]['name']}",
            line_color=CLUSTER_METADATA[cname]['color']
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        ),
        showlegend=True, template="plotly_dark",
        height=520, margin=dict(l=40, r=40, t=30, b=30)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### Feature Distribution Explorer")
    feat_sel3 = st.selectbox(
        "Inspect feature distribution across clusters:",
        FEATURE_NAMES, format_func=lambda f: FEATURE_LABELS[f], key="feat_sel_s3"
    )
    fig_box = px.box(
        df, x='kmeans_cluster', y=feat_sel3, color='kmeans_cluster',
        color_discrete_map={'Cluster 1': '#00d4ff', 'Cluster 2': '#ff9f1c', 'Cluster 3': '#ff007f'},
        points='suspectedoutliers',
        title=f"Cluster Distribution: {FEATURE_LABELS[feat_sel3]}",
        template="plotly_dark"
    )
    fig_box.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Raw Centroid Values Table")
    st.dataframe(centroids_raw.round(2).reset_index(), hide_index=True, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SELF-ORGANIZING MAP (SOM)
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("4."):
    st.subheader("4. Self-Organizing Map (SOM): Topology Mapping & Cluster Collapse")

    st.markdown("""
    We implemented a **5 × 5 Kohonen SOM from scratch using NumPy** with 100 epochs,
    decaying learning rate, and a Gaussian neighbourhood function.
    """)

    col_som1, col_som2 = st.columns([0.5, 0.5])

    with col_som1:
        st.markdown("#### SOM Architecture Parameters")
        som_params = pd.DataFrame([
            {"Parameter": "Lattice Topology",          "Value": "2D Rectangular (5 × 5 = 25 neurons)"},
            {"Parameter": "Input Dimension",            "Value": "8 unscaled continuous variables"},
            {"Parameter": "Training Epochs",            "Value": "100 iterations"},
            {"Parameter": "Initial Learning Rate (η₀)", "Value": "0.5 (exponential decay)"},
            {"Parameter": "Initial Radius (σ₀)",        "Value": "2.5 (decays with τ = T / ln(σ₀))"},
            {"Parameter": "Neighbourhood Function",     "Value": "Gaussian h(t) = exp(−d² / 2σ²)"},
            {"Parameter": "Macro-clustering Strategy",  "Value": "Nearest centroid mapping (k=3)"}
        ])
        st.dataframe(som_params, hide_index=True, use_container_width=True)

    with col_som2:
        st.markdown("#### Macro-Cluster Results vs. Stanford Paper Target")
        som_comp = pd.DataFrame([
            {"Cluster": "SOM Cluster 1", "Our Size": 1045, "Paper Target": "~1079", "Status": "✅ Recovered"},
            {"Cluster": "SOM Cluster 2", "Our Size": 594,  "Paper Target": "~517",  "Status": "⚠️ Slightly Shifted"},
            {"Cluster": "SOM Cluster 3", "Our Size": 0,    "Paper Target": "~43",   "Status": "❌ Absorbed"}
        ])
        st.dataframe(som_comp, hide_index=True, use_container_width=True)
        st.metric("SOM Silhouette Score", "0.4024", delta="−0.0389 vs paper target 0.44")

    st.markdown("---")

    # ── SOM vs K-Means Boomerang Side-by-Side ────────────────────────────
    st.markdown("#### 📊 SOM vs K-Means — Boomerang Manifold Comparison")
    st.caption("Comparing how K-Means and SOM macro-clusters map onto the PCA Boomerang. Notice the 3rd cluster (magenta) is absent from the SOM plot.")

    col_km_plot, col_som_plot = st.columns(2)

    with col_km_plot:
        fig_km_bm = px.scatter(
            df, x='PC1', y='PC2', color='kmeans_cluster',
            color_discrete_map={'Cluster 1': '#00d4ff', 'Cluster 2': '#ff9f1c', 'Cluster 3': '#ff007f'},
            opacity=0.6, title="K-Means (3 clusters — all recovered)",
            hover_data=['mean_dmsnr_curve', 'skewness_dmsnr_curve'],
            template="plotly_dark"
        )
        fig_km_bm.update_traces(marker=dict(size=5))
        fig_km_bm.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_km_bm, use_container_width=True)

    with col_som_plot:
        if 'som_cluster' in df.columns:
            fig_som_bm = px.scatter(
                df, x='PC1', y='PC2', color='som_cluster',
                color_discrete_map={'SOM Cluster 1': '#00d4ff', 'SOM Cluster 2': '#ff9f1c'},
                opacity=0.6, title="SOM Macro-clusters (Cluster 3 absorbed — only 2 regions)",
                hover_data=['mean_dmsnr_curve', 'skewness_dmsnr_curve'],
                template="plotly_dark"
            )
            fig_som_bm.update_traces(marker=dict(size=5))
            fig_som_bm.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_som_bm, use_container_width=True)
        else:
            st.warning("SOM labels not found. Run `som_model.py` to generate `som_clustered_pulsars.csv`.")

    st.markdown("---")
    st.markdown("#### 🔬 Why Did Cluster 3 Collapse?")
    st.warning("""
    **Mechanism of Macro-Cluster Collapse:**
    1. **Neuron-to-Centroid Assignment:** After training, all 25 neuron weight vectors are assigned to the nearest K-Means centroid (k=3).
    2. **Sparse Geometric Location:** Cluster 3 holds only 44 pulsars (2.7% of data) at an extreme corner of feature space (`skewness_dmsnr_curve = 255.6`).
    3. **Lattice Inertia:** No SOM neuron converged close enough to that sparse centroid. All 25 neurons mapped to Cluster 1 or Cluster 2.
    4. **Divergence from Reference Paper:** The original TensorFlow SOM collapsed to 3 active grid cells directly due to implementation-specific seed behavior — our NumPy SOM keeps all 25 neurons active.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LIVE PULSAR SANDBOX
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("5."):
    st.subheader("5. Interactive Pulsar Candidate Sandbox")
    st.markdown("Pick a real pulsar, choose a preset archetype, or tune sliders to see live projection onto the Boomerang Manifold and instant cluster assignment.")

    col_ctrl, col_plot = st.columns([0.35, 0.65])

    # Pre-compute ordered centroids (explicit index order — fixes bug #4)
    centroids_df  = df.groupby('kmeans_cluster')[FEATURE_NAMES].mean().loc[
        ['Cluster 1', 'Cluster 2', 'Cluster 3']
    ]
    cluster_names = centroids_df.index.tolist()
    centroids     = centroids_df.values.astype(np.float64)

    with col_ctrl:
        st.markdown("#### 🧪 Select Candidate")
        mode = st.radio("Source:", ["Real Pulsar by Index", "Preset Archetypes", "Manual Sliders"])

        test_vector  = None
        sample_label = "Custom Candidate"

        if mode == "Real Pulsar by Index":
            col_idx, col_rand = st.columns([0.75, 0.25])
            with col_idx:
                idx = st.slider("Pulsar Index (0–1638):", 0, len(df)-1,
                                st.session_state.get("sandbox_idx", 142),
                                key="sandbox_idx")
            with col_rand:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🎲", help="Pick a random pulsar"):
                    st.session_state["sandbox_idx"] = int(np.random.randint(0, len(df)))
                    st.rerun()

            row = df.iloc[st.session_state.get("sandbox_idx", idx)]
            test_vector  = np.asarray(row[FEATURE_NAMES].values, dtype=np.float64)
            sample_label = f"Pulsar #{st.session_state.get('sandbox_idx', idx)} ({row['kmeans_cluster']})"
            st.caption(f"Actual assigned cluster: **{row['kmeans_cluster']}**")

        elif mode == "Preset Archetypes":
            preset = st.selectbox("Choose Archetype:", [
                "Canonical Pulsar (Cluster 1 Typical)",
                "Distant High-DM Pulsar (Cluster 2 Typical)",
                "Nearby Millisecond Pulsar (Cluster 3 Extreme)"
            ])
            preset_vecs = {
                "Canonical Pulsar (Cluster 1 Typical)":      np.array([69.2, 39.2,  2.4, 10.6,  20.6, 48.5,  3.7,  18.8], dtype=np.float64),
                "Distant High-DM Pulsar (Cluster 2 Typical)":np.array([34.4, 37.4,  4.4, 24.3,  99.1, 72.1,  0.4,  -0.3], dtype=np.float64),
                "Nearby Millisecond Pulsar (Cluster 3 Extreme)": np.array([94.9, 45.2, 0.9, 2.4, 1.3, 12.8, 14.5, 255.6], dtype=np.float64)
            }
            test_vector  = preset_vecs[preset]
            sample_label = preset

        else:  # Manual Sliders
            st.markdown("##### Key Driving Features:")
            ip_mean  = st.slider("Mean Pulse Profile (Brightness):", 10.0, 150.0, 70.0)
            dm_mean  = st.slider("Mean DM-SNR (Signal Strength):",    0.0,  150.0, 25.0)
            dm_skew  = st.slider("Skewness DM-SNR (Tail Asymmetry):", -2.0, 300.0, 20.0)
            test_vector  = np.array([ip_mean, 40.0, 2.5, 12.0, dm_mean, 50.0, 4.0, dm_skew], dtype=np.float64)
            sample_label = "Custom Simulated Signal"

        # ── Classification ──────────────────────────────────────────────
        distances       = np.linalg.norm(centroids - test_vector, axis=1)
        pred_cluster    = cluster_names[int(np.argmin(distances))]
        meta            = CLUSTER_METADATA[pred_cluster]

        st.markdown("---")
        st.markdown("#### 🎯 Model Classification")
        st.markdown(f"""
        <div class="stat-card" style="border: 2px solid {meta['color']};">
            <span style="color:{meta['color']}; font-weight:700; font-size:1.2rem;">{pred_cluster}</span><br>
            <span style="font-size:0.95rem; font-weight:600; color:#f8fafc;">{meta['name']}</span>
            <p style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">{meta['astronomy_insight']}</p>
            <div style="font-size:0.8rem; color:#64748b;">
                Distance to centroid: <b>{distances[cluster_names.index(pred_cluster)]:.1f}</b> units
            </div>
        </div>""", unsafe_allow_html=True)

        # Distance bar to all 3 centroids
        dist_fig = px.bar(
            x=cluster_names, y=distances,
            color=cluster_names,
            color_discrete_map={'Cluster 1': '#00d4ff', 'Cluster 2': '#ff9f1c', 'Cluster 3': '#ff007f'},
            labels={'x': '', 'y': 'Euclidean Distance'},
            title="Distance to Each Cluster Centroid",
            template="plotly_dark"
        )
        dist_fig.update_layout(height=220, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(dist_fig, use_container_width=True)

    with col_plot:
        test_pca = pca_model.transform(test_vector.reshape(1, -1))[0]

        fig_sb = px.scatter(
            df, x='PC1', y='PC2', color='kmeans_cluster',
            color_discrete_map={'Cluster 1': '#00d4ff', 'Cluster 2': '#ff9f1c', 'Cluster 3': '#ff007f'},
            opacity=0.4,
            title=f"Live Projection: {sample_label}",
            hover_data=['mean_integrated_profile', 'mean_dmsnr_curve', 'skewness_dmsnr_curve'],
            template="plotly_dark"
        )
        fig_sb.add_trace(go.Scatter(
            x=[test_pca[0]], y=[test_pca[1]],
            mode='markers+text',
            name="Target Candidate",
            text=[f"📍 {sample_label}"],
            textposition="top center",
            marker=dict(
                color='#ffffff', size=18, symbol='star',
                line=dict(color=meta['color'], width=3)
            )
        ))
        fig_sb.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_sb, use_container_width=True)
        st.caption(f"Projected coordinates: **PC1 = {test_pca[0]:.2f}**, **PC2 = {test_pca[1]:.2f}**")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PAPER PARITY & FINAL SCORECARD
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("6."):
    st.subheader("6. Parity Verification — Debesai, Gutierrez & Koyluoglu (Stanford CS 229)")

    st.markdown("""
    Verification of our pipeline against **Table 2 and Table 3** of the reference paper.
    Each row is colored green (✅ match) or amber (⚠️ partial deviation).
    """)

    # ── Color-coded scorecard rows ────────────────────────────────────────
    for row in PAPER_TARGETS['scorecard']:
        icon    = "✅" if row['Match'] else "⚠️"
        css_cls = "score-row-pass" if row['Match'] else "score-row-warn"
        st.markdown(f"""
        <div class="{css_cls}">
            <span style="font-weight:700;">{icon} {row['Metric / Component']}</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Paper: <code>{row['Paper Target']}</code>
            &nbsp;&nbsp;→&nbsp;&nbsp;
            Ours: <code>{row['Our Result']}</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Full Centroid Comparison Tables")

    tab_km, tab_som = st.tabs(["K-Means Centroids", "SOM Centroids"])

    with tab_km:
        our_km = df.groupby('kmeans_cluster')[FEATURE_NAMES].mean().loc[
            ['Cluster 1', 'Cluster 2', 'Cluster 3']].round(2).reset_index()
        paper_km = pd.DataFrame(PAPER_TARGETS['kmeans']['clusters'])
        paper_km[FEATURE_NAMES] = pd.DataFrame(
            paper_km['centroid'].tolist(), columns=FEATURE_NAMES
        )
        paper_km = paper_km.drop(columns=['centroid']).rename(columns={'cluster': 'kmeans_cluster'})
        st.markdown("**Our K-Means Centroids**")
        st.dataframe(our_km, hide_index=True, use_container_width=True)
        st.markdown("**Paper Target Centroids (Table 2)**")
        st.dataframe(paper_km.round(2), hide_index=True, use_container_width=True)

    with tab_som:
        paper_som = pd.DataFrame(PAPER_TARGETS['som']['clusters'])
        paper_som[FEATURE_NAMES] = pd.DataFrame(
            paper_som['centroid'].tolist(), columns=FEATURE_NAMES
        )
        paper_som = paper_som.drop(columns=['centroid'])
        st.markdown("**Paper Target SOM Centroids (Table 3)**")
        st.dataframe(paper_som.round(3), hide_index=True, use_container_width=True)
        st.info("Our SOM Silhouette Score: **0.4024** (target 0.44). The 3rd macro-cluster was absorbed — see Section 4 for analysis.")

    st.markdown("---")
    st.markdown("#### 🎓 Defense Summary & Conclusions")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.success("""
        **What We Successfully Proved:**
        - **K-Means Replicability:** Exact cluster sizes (974, 621, 44) and Silhouette Score 0.4413 (~0.44 target, +0.3% deviation).
        - **Manifold Validation:** PCA confirms 2 components explain 86% of cosmic variance, reproducing the Boomerang arc.
        - **Feature Drivers Confirmed:** `skewness_dmsnr_curve` and `mean_dmsnr_curve` are the dominant variance axes.
        """)
    with col_c2:
        st.info("""
        **Key Astrophysical Takeaways:**
        - **Cluster 1 (974):** Canonical rotation-powered pulsars — the bulk galactic population.
        - **Cluster 2 (621):** Distant, heavily dispersed objects deeper in the galactic plane.
        - **Cluster 3 (44):** Nearby pulsars with extreme DM asymmetry; millisecond or recycled candidates.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    Categorization of Radio Pulsar Signals &nbsp;·&nbsp; Unsupervised Machine Learning &nbsp;·&nbsp; SEM-5 Defense
</div>
""", unsafe_allow_html=True)
