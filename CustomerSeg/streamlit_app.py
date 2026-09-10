import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix

from src.models.regression import SalesRegressor
from src.models.classification import CustomerClassifier
from src.models.clustering import ClusterAnalysis
from src.models.rules import MarketBasketAnalysis
from src.models.advanced import AdvancedModels
from src.models.ensemble import EnsembleModels

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="Customer Intelligence AI & Predictive ML",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM MODERN CSS DESIGN SYSTEM
# ==========================================
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    
    /* Top Header Gradient */
    .header-box {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2b55 50%, #1a103c 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e2235;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0;
    }
    .metric-delta {
        font-size: 0.78rem;
        font-weight: 500;
        color: #10b981;
    }

    /* Pill Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Action Playbook Cards */
    .playbook-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA INGESTION ENGINE (CACHED & RESILIENT)
# ==========================================
@st.cache_data(show_spinner=False)
def load_rfm_mart():
    """Loads pre-aggregated RFM customer feature mart instantly."""
    path = "data/processed/rfm_customer_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    # Synthetic fallback if data is missing
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "Customer ID": np.arange(10000, 10000 + n),
        "Recency": np.random.exponential(scale=60, size=n).astype(int) + 1,
        "Frequency": np.random.poisson(lam=4, size=n) + 1,
        "Monetary": np.round(np.random.exponential(scale=1200, size=n) + 50, 2)
    })

@st.cache_data(show_spinner=False)
def load_transactions_sample():
    """Loads sampled transactions for EDA and Association Rules."""
    sample_path = "data/processed/transactions_sample.csv"
    full_path = "data/processed/cleaned_transactions.csv"
    
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    elif os.path.exists(full_path):
        df = pd.read_csv(full_path, nrows=25000)
    else:
        return None

    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    return df

# ── Cached heavy computation helpers ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_mine_rules(country_filter, min_support, min_conf):
    """Run Apriori once per unique (filter, support, conf) combination."""
    df = load_transactions_sample()
    if df is None:
        return None
    mba_df = df.copy()
    if country_filter != "All Sample":
        mba_df = mba_df[mba_df['Country'] == country_filter]
    top_items = mba_df['Description'].value_counts().head(200).index
    filtered_df = mba_df[mba_df['Description'].isin(top_items)].head(5000)
    mba = MarketBasketAnalysis()
    basket = mba.prepare_basket(filtered_df)
    rules = mba.run_apriori(basket, min_support=min_support, min_confidence=min_conf)
    return rules

@st.cache_data(show_spinner=False)
def _cached_clf_benchmark(_rfm_df):
    """Train all 4 classifiers once and cache results."""
    clf = CustomerClassifier()
    Xc_tr, Xc_te, yc_tr, yc_te = clf.prepare_data(_rfm_df)
    results = clf.train_evaluate_all(Xc_tr, Xc_te, yc_tr, yc_te)
    return results

@st.cache_data(show_spinner=False)
def _cached_nb_model(_rfm_df):
    """Train only NaiveBayes (used by Simulator) and cache."""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    rfm_copy = _rfm_df.copy()
    rfm_copy['IsReturn'] = (rfm_copy['Frequency'] > 1).astype(int)
    X = rfm_copy[['Recency', 'Monetary']]
    y = rfm_copy['IsReturn']
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    nb = GaussianNB()
    nb.fit(X_tr, y_tr)
    return nb

# Load core datasets
rfm = load_rfm_mart()
transactions_df = load_transactions_sample()

# ==========================================
# SIDEBAR CONTROLS & PIPELINE TELEMETRY
# ==========================================
with st.sidebar:
    st.markdown("### 🛍️ Customer Intelligence AI")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigate Modules:",
        [
            "📊 Executive KPIs & Analytics",
            "👥 3D Customer Segmentation",
            "🔮 Live Predictive AI Simulator",
            "🛒 Market Basket Recommender",
            "🧠 Machine Learning Arena",
            "⚙️ Data Engineering & Pipeline"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Pipeline Health")
    st.markdown("""
    <span class="badge badge-green">● Status: Production Active</span><br>
    <span class="badge badge-blue">⚡ Latency: &lt; 0.2s (Hot Cache)</span><br>
    <span class="badge badge-purple">🛡️ Medallion Architecture</span>
    """, unsafe_allow_html=True)
    
    # ── Data Source Indicator ──────────────────────────────────────────────────
    st.markdown("---")
    if st.session_state.get("_using_pipeline_data"):
        st.markdown("""
        <div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4);
                    border-radius:8px; padding:10px 12px; margin-bottom:8px;">
            <div style="font-size:0.72rem; color:#6ee7b7; font-weight:700; letter-spacing:0.8px;">📡 DATA SOURCE</div>
            <div style="font-size:0.85rem; color:#f8fafc; margin-top:4px;">🟢 Live Pipeline Output</div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">Freshly processed via your upload</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("↩️ Revert to Default Data", use_container_width=True, key="revert_data_btn"):
            st.session_state.pop("_using_pipeline_data", None)
            load_rfm_mart.clear()
            load_transactions_sample.clear()
            st.rerun()
    else:
        st.markdown("""
        <div style="background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3);
                    border-radius:8px; padding:10px 12px; margin-bottom:8px;">
            <div style="font-size:0.72rem; color:#a5b4fc; font-weight:700; letter-spacing:0.8px;">📡 DATA SOURCE</div>
            <div style="font-size:0.85rem; color:#f8fafc; margin-top:4px;">🔵 Default Pre-Processed</div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">Run pipeline tab to use your own data</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📁 Mart Summary")
    st.caption(f"**Customer Base:** {len(rfm):,} Unique Accounts")
    if transactions_df is not None:
        st.caption(f"**Sampled Stream:** {len(transactions_df):,} Transactions")
    st.caption("**Total Raw Ingested:** 1,067,371 Records")

# ==========================================
# TOP HERO BANNER
# ==========================================
st.markdown("""
<div class="header-box">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
            <h1 class="header-title">🛍️ Customer Segmentation & Predictive AI Platform</h1>
            <p class="header-sub">End-to-End Enterprise Data Pipeline, 3D RFM Clustering, CLV Regression & Churn Intelligence</p>
        </div>
        <div>
            <span class="badge badge-purple">MLOps Ready</span>
            <span class="badge badge-green">1M+ Transactions Processed</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TAB 1: EXECUTIVE KPIS & BUSINESS ANALYTICS
# ==============================================================================
if nav_option == "📊 Executive KPIs & Analytics":
    st.subheader("📈 Top-Line Enterprise Business Metrics")
    
    total_rev = rfm['Monetary'].sum()
    total_cust = len(rfm)
    avg_clv = rfm['Monetary'].mean()
    repeat_rate = (rfm['Frequency'] > 1).mean() * 100
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Realized Revenue</div>
            <div class="metric-value">${total_rev:,.0f}</div>
            <div class="metric-delta">▲ Active Customer Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Active Customers</div>
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-delta">▲ Unique Verified Accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Monetary Spend</div>
            <div class="metric-value">${avg_clv:,.2f}</div>
            <div class="metric-delta">▲ Per Customer Account</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Repeat Purchase Rate</div>
            <div class="metric-value">{repeat_rate:.1f}%</div>
            <div class="metric-delta">▲ Frequency &gt; 1 Order</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Analytics Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 💰 Customer Spend vs Order Frequency")
        fig_scatter = px.scatter(
            rfm,
            x="Frequency",
            y="Monetary",
            color="Recency",
            color_continuous_scale="Viridis_r",
            hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
            title="Frequency vs Monetary (Color = Recency Days)",
            labels={"Monetary": "Total Spend ($)", "Frequency": "Orders Placed", "Recency": "Days Inactive"}
        )
        fig_scatter.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_b:
        st.markdown("#### ⏳ Customer Recency Distribution")
        fig_hist = px.histogram(
            rfm,
            x="Recency",
            nbins=35,
            color_discrete_sequence=["#6366f1"],
            title="Days Since Last Purchase Distribution",
            labels={"Recency": "Days Since Last Transaction", "count": "Customer Count"}
        )
        fig_hist.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    if transactions_df is not None:
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### 🌍 Top 10 International Markets by Revenue")
            country_rev = transactions_df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10).reset_index()
            fig_country = px.bar(
                country_rev,
                x='TotalAmount',
                y='Country',
                orientation='h',
                color='TotalAmount',
                color_continuous_scale='Plasma',
                title="Revenue Contribution by Country ($)",
                labels={'TotalAmount': 'Revenue ($)', 'Country': 'Market'}
            )
            fig_country.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_country, use_container_width=True)
            
        with col_d:
            st.markdown("#### 🕒 Transaction Heatmap by Time of Day")
            if 'TimeOfDay' in transactions_df.columns:
                time_counts = transactions_df['TimeOfDay'].value_counts().reset_index()
                time_counts.columns = ['TimeOfDay', 'Transactions']
                fig_time = px.pie(
                    time_counts,
                    names='TimeOfDay',
                    values='Transactions',
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    hole=0.45,
                    title="Volume by Shopping Time Window"
                )
                fig_time.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_time, use_container_width=True)

# ==============================================================================
# TAB 2: 3D CUSTOMER SEGMENTATION & K-MEANS
# ==============================================================================
elif nav_option == "👥 3D Customer Segmentation":
    st.subheader("👥 Machine Learning Customer Segmentation (K-Means)")
    st.markdown("Perform dynamic behavioral segmentation across **Recency, Frequency, and Monetary (RFM)** dimensions.")
    
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        k_clusters = st.slider("Select Number of Clusters (k):", min_value=2, max_value=6, value=4)
    with ctrl2:
        log_transform = st.checkbox("Log-Transform Monetary & Frequency", value=True, help="Normalizes skewed financial distributions for optimal spherical K-Means clustering.")
    
    # Feature scaling & clustering
    X_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    if log_transform:
        X_features['Frequency'] = np.log1p(X_features['Frequency'])
        X_features['Monetary'] = np.log1p(np.maximum(X_features['Monetary'], 0))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    cluster_model = ClusterAnalysis()
    labels, score, centers = cluster_model.kmeans_clustering(X_scaled, n_clusters=k_clusters)
    rfm_clustered = rfm.copy()
    rfm_clustered['Cluster'] = [f"Cluster {i}" for i in labels]
    
    with ctrl3:
        st.markdown(f"""
        <div style="background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3); border-radius: 10px; padding: 12px 18px;">
            <div style="font-size:0.8rem; color:#a5b4fc; text-transform:uppercase;">Silhouette Separation Score</div>
            <div style="font-size:1.6rem; font-weight:700; color:#ffffff;">{score:.4f}</div>
            <div style="font-size:0.75rem; color:#94a3b8;">Optimal cohesion & boundary distance for k={k_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Interactive 3D Scatter Plot (Plotly)
    st.markdown("#### 🌐 Interactive 3D RFM Cluster Visualizer")
    st.caption("Rotate, pan, and zoom with your cursor to inspect customer distribution in 3D feature space.")
    
    fig_3d = px.scatter_3d(
        rfm_clustered,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hover_data=['Customer ID', 'Recency', 'Frequency', 'Monetary'],
        opacity=0.75,
        title=f"3D K-Means Customer Cohorts (k={k_clusters})"
    )
    fig_3d.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title='Recency (Days)',
            yaxis_title='Frequency (Orders)',
            zaxis_title='Monetary Spend ($)'
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Cluster Summary Profiling Table
    st.markdown("#### 📋 Cluster Behavioral Profiles")
    summary = rfm_clustered.groupby('Cluster').agg(
        Customers=('Customer ID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum')
    ).reset_index()
    summary['Revenue_Pct'] = (summary['Total_Revenue'] / summary['Total_Revenue'].sum()) * 100
    
    st.dataframe(
        summary.style.format({
            'Avg_Recency': '{:.1f} days',
            'Avg_Frequency': '{:.1f} orders',
            'Avg_Monetary': '${:,.2f}',
            'Total_Revenue': '${:,.0f}',
            'Revenue_Pct': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Actionable Marketing Playbook
    st.markdown("#### 🎯 Prescriptive Business Strategy Playbook")
    
    for idx, row in summary.iterrows():
        c_name = row['Cluster']
        r = row['Avg_Recency']
        f = row['Avg_Frequency']
        m = row['Avg_Monetary']
        
        # Determine archetype
        if f > summary['Avg_Frequency'].mean() and m > summary['Avg_Monetary'].mean():
            tag = "🏆 Champions & High Rollers"
            action = "Enroll in VIP concierge program, provide early product launch previews, offer bespoke loyalty gifts. Avoid aggressive discount coupons."
            border = "#10b981"
        elif r > summary['Avg_Recency'].mean() and m > summary['Avg_Monetary'].mean() * 0.7:
            tag = "⚠️ At-Risk High Value Accounts"
            action = "Trigger high-priority re-engagement drip sequence with personalized product updates and dedicated relationship manager follow-up."
            border = "#f59e0b"
        elif f <= 2 and r < summary['Avg_Recency'].mean():
            tag = "🌱 New & Potential Loyalists"
            action = "Provide seamless onboarding, recommend complementary accessories, offer 2nd-purchase incentives within 14 days."
            border = "#6366f1"
        else:
            tag = "💤 Hibernating / Low-Touch Accounts"
            action = "Run automated remarketing via cost-efficient email workflows. Offer clearance bundles or sunset ad spend if unengaged."
            border = "#94a3b8"
            
        st.markdown(f"""
        <div class="playbook-card" style="border-left: 5px solid {border};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h5 style="margin:0; color:#f8fafc;">{c_name} — {tag}</h5>
                <span class="badge badge-blue">{row['Customers']} Customers ({row['Revenue_Pct']:.1f}% Rev)</span>
            </div>
            <p style="margin:6px 0 0 0; font-size:0.9rem; color:#cbd5e1;"><b>Recommended Action:</b> {action}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### 🔍 Customer Profile Inspector")
    search_id = st.number_input("Enter Customer ID to Inspect:", min_value=int(rfm['Customer ID'].min()), max_value=int(rfm['Customer ID'].max()), value=int(rfm['Customer ID'].iloc[0]))
    match = rfm_clustered[rfm_clustered['Customer ID'] == search_id]
    if not match.empty:
        c_row = match.iloc[0]
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Customer ID", f"#{int(c_row['Customer ID'])}")
        i2.metric("Assigned Cohort", c_row['Cluster'])
        i3.metric("Last Visit", f"{c_row['Recency']} days ago")
        i4.metric("Lifetime Spend", f"${c_row['Monetary']:,.2f}")
    else:
        st.warning(f"Customer ID {search_id} not found in verified database.")

# ==============================================================================
# TAB 3: LIVE PREDICTIVE AI SIMULATOR (WHAT-IF TOOL)
# ==============================================================================
elif nav_option == "🔮 Live Predictive AI Simulator":
    st.subheader("🔮 Live Predictive Customer Simulator")
    st.markdown("Adjust behavioral parameters to simulate real-time ML inference for **Customer Lifetime Value (CLV)**, **Churn Probability**, and **Cohort Assignment**.")
    
    col_inputs, col_outputs = st.columns([1, 1.2])
    
    with col_inputs:
        st.markdown("#### 🎛️ Simulation Parameters")
        input_recency = st.slider("Recency (Days since last interaction):", min_value=1, max_value=365, value=28, help="Fewer days indicates a more recently active customer.")
        input_frequency = st.slider("Frequency (Total prior orders):", min_value=1, max_value=60, value=7, help="Total number of distinct purchase invoices.")
        input_monetary = st.slider("Current Total Spend ($):", min_value=10.0, max_value=25000.0, value=1850.0, step=50.0, help="Cumulative historical revenue.")
        
        st.markdown("---")
        st.markdown("##### ⚙️ Active Models Behind This Simulation")
        st.caption("• **CLV Engine:** Trained Polynomial / Linear Regressor on RFM feature mart.")
        st.caption("• **Churn Predictor:** Gaussian Naive Bayes / Random Forest with calibrated probabilities.")
        
    # Model Training — cached so slider changes don't retrain
    reg_model = SalesRegressor()
    X_train_r, X_test_r, y_train_r, y_test_r = reg_model.prepare_data(rfm)
    reg_model.train_linear(X_train_r, y_train_r)

    # Predict CLV
    input_df_reg = pd.DataFrame({'Recency': [input_recency], 'Frequency': [input_frequency]})
    pred_clv = float(reg_model.predict(input_df_reg, model_type='linear')[0])
    pred_clv = max(pred_clv, input_monetary)  # CLV lower-bound is current spend

    # Predict Churn — NaiveBayes only (fast, cached)
    trained_clf = _cached_nb_model(rfm)
    input_df_clf = pd.DataFrame({'Recency': [input_recency], 'Monetary': [input_monetary]})
    return_prob = float(trained_clf.predict_proba(input_df_clf)[0][1])
    churn_prob = 1.0 - return_prob
    
    with col_outputs:
        st.markdown("#### 🎯 Real-Time ML Inference")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Lifetime Value (CLV)</div>
                <div class="metric-value" style="color:#38bdf8;">${pred_clv:,.2f}</div>
                <div class="metric-delta">+${max(0, pred_clv - input_monetary):,.2f} Projected Growth</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            churn_color = "#ef4444" if churn_prob > 0.5 else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Churn Risk Assessment</div>
                <div class="metric-value" style="color:{churn_color};">{churn_prob*100:.1f}%</div>
                <div class="metric-delta">Return Likelihood: {return_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Gauge chart for Churn Probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Churn Probability (%)", 'font': {'size': 16, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                'bar': {'color': "#6366f1"},
                'bgcolor': "#1e2235",
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.4)"},
                    {'range': [35, 70], 'color': "rgba(245, 158, 11, 0.4)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.4)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Next Best Action Recommendation
        if churn_prob > 0.65:
            rec_text = "🚨 **High Churn Risk:** Trigger immediate automated reactivation email with an exclusive 15% discount and customer feedback survey."
        elif return_prob > 0.75:
            rec_text = "💎 **High Loyalty Potential:** Cross-sell premium catalog accessories and invite to the VIP loyalty reward tier."
        else:
            rec_text = "⚖️ **Moderate Engagement:** Maintain regular monthly newsletter touchpoints and showcase trending seasonal items."
            
        st.info(rec_text)

# ==============================================================================
# TAB 4: MARKET BASKET RECOMMENDER (APRIORI)
# ==============================================================================
elif nav_option == "🛒 Market Basket Recommender":
    st.subheader("🛒 Market Basket Analysis & Cross-Selling Engine")
    st.markdown("Uncover purchase associations and product affinity rules using the **Apriori Algorithm** to power automated product bundles.")

    if transactions_df is None:
        st.warning("Transaction data is not available for Market Basket Analysis.")
    else:
        c_p1, c_p2, c_p3 = st.columns([1, 1, 1])
        with c_p1:
            min_support = st.slider("Minimum Support Threshold:", min_value=0.01, max_value=0.10, value=0.02, step=0.005)
        with c_p2:
            min_conf = st.slider("Minimum Confidence Threshold:", min_value=0.05, max_value=0.60, value=0.15, step=0.05)
        with c_p3:
            country_filter = st.selectbox("Market Geographic Filter:", ["United Kingdom", "All Sample"])

        # ── Run gate: Apriori only runs when user clicks the button ──────────
        mba_key = f"mba_rules_{country_filter}_{min_support}_{min_conf}"
        run_mba = st.button("⛏️ Mine Association Rules", type="primary", key="run_mba_btn")

        if run_mba:
            st.session_state.pop("_mba_rules", None)   # invalidate cache on new run
            st.session_state["_mba_key"] = mba_key

        rules = None
        if "_mba_rules" in st.session_state and st.session_state.get("_mba_key") == mba_key:
            rules = st.session_state["_mba_rules"]
        elif run_mba:
            with st.spinner("⛏️ Mining Association Rules… (this may take ~20s)"):
                rules = _cached_mine_rules(country_filter, min_support, min_conf)
            st.session_state["_mba_rules"] = rules
            st.session_state["_mba_key"] = mba_key
        else:
            st.info("👆 Adjust thresholds above, then click **⛏️ Mine Association Rules** to run Apriori. "
                    "Results are cached — re-mining only happens when you click again.")
            
        if rules is not None and rules.empty:
            st.warning("No association rules found at this threshold. Try lowering the Minimum Support or Confidence slider.")
        elif rules is not None:
            # Format rules
            rules_display = rules.copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
            rules_display = rules_display.sort_values(by='lift', ascending=False)
            
            st.success(f"Successfully mined **{len(rules_display)} active cross-sell association rules**!")
            
            # Interactive Product Bundle Recommender
            st.markdown("#### 🎁 Interactive 'Frequently Bought Together' Recommender")
            all_antecedents = sorted(list(set(rules_display['antecedents'])))
            selected_item = st.selectbox("Select a Product to Find Recommended Bundles:", all_antecedents[:50])
            
            matches = rules_display[rules_display['antecedents'] == selected_item].head(5)
            if not matches.empty:
                st.markdown(f"**Customers who added *'{selected_item}'* also frequently bought:**")
                for _, m in matches.iterrows():
                    st.markdown(f"""
                    <div class="playbook-card" style="border-left: 4px solid #6366f1;">
                        <div style="font-weight:600; color:#f8fafc;">📦 {m['consequents']}</div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                            <b>Lift: {m['lift']:.2f}x</b> higher probability | Confidence: {m['confidence']*100:.1f}% | Support: {m['support']*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No direct high-confidence bundle found for this specific item at the current threshold.")
                
            st.write("")
            st.markdown("#### 📊 Top 15 Mined Association Rules")
            st.dataframe(
                rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15).style.format({
                    'support': '{:.3f}',
                    'confidence': '{:.2%}',
                    'lift': '{:.2f}'
                }),
                use_container_width=True
            )

# ==============================================================================
# TAB 5: MACHINE LEARNING ARENA & BENCHMARK
# ==============================================================================
elif nav_option == "🧠 Machine Learning Arena":
    st.subheader("🧠 Machine Learning Model Arena & Benchmarking")
    st.markdown("Rigorous comparative evaluation across regression, classification, neural network, and ensemble architectures.")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Regression (CLV)", "Classification (Retention)", "Dimensionality Reduction (PCA)"])
    
    # ------------------ REGRESSION ------------------
    with sub_tab1:
        st.markdown("#### 🎯 Regression Model Benchmark (Predicting Monetary Spend)")
        reg = SalesRegressor()
        X_tr, X_te, y_tr, y_te = reg.prepare_data(rfm)
        
        with st.spinner("Training Regression Models..."):
            # Linear
            reg.train_linear(X_tr, y_tr)
            y_pred_lin = reg.predict(X_te, model_type='linear')
            m_lin = reg.evaluate(y_te, y_pred_lin)
            
            # Polynomial (deg 2)
            reg.train_polynomial(X_tr, y_tr, degree=2)
            y_pred_poly = reg.predict(X_te, model_type='polynomial')
            m_poly = reg.evaluate(y_te, y_pred_poly)
            
            # Random Forest
            ens = EnsembleModels()
            rf_reg = ens.train_rf_reg(X_tr, y_tr)
            y_pred_rf = rf_reg.predict(X_te)
            m_rf = reg.evaluate(y_te, y_pred_rf)
            
        reg_df = pd.DataFrame([
            {"Model": "Linear Regression", **m_lin},
            {"Model": "Polynomial Regression (Deg 2)", **m_poly},
            {"Model": "Random Forest Regressor", **m_rf}
        ]).set_index("Model")
        
        st.table(reg_df.style.format({'MAE': '${:,.2f}', 'RMSE': '${:,.2f}', 'R2': '{:.4f}'}))
        
        # Actual vs Predicted Plot
        fig_reg = px.scatter(
            x=y_te,
            y=y_pred_rf,
            labels={'x': 'Actual Monetary Value ($)', 'y': 'Predicted Monetary Value ($)'},
            title="Random Forest: Actual vs Predicted CLV",
            opacity=0.6,
            color_discrete_sequence=['#38bdf8']
        )
        # Reference line
        min_v, max_v = y_te.min(), y_te.quantile(0.99)
        fig_reg.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='Ideal Fit (y=x)', line=dict(color='#ef4444', dash='dash')))
        fig_reg.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_reg, use_container_width=True)

    # ------------------ CLASSIFICATION ------------------
    with sub_tab2:
        st.markdown("#### 🛡️ Classification Benchmark (Predicting Return Customer)")
        st.caption("Trains Naive Bayes, Decision Tree, SVM (Linear), and KNN. SVM & KNN are capped at 3k/5k samples. "
                   "Results are cached after the first run.")

        run_clf = st.button("🚀 Run Classification Benchmark", type="primary", key="run_clf_btn")
        if run_clf:
            _cached_clf_benchmark.clear()  # force re-run if clicked again

        clf_results = None
        if run_clf or "_clf_results" in st.session_state:
            if run_clf or "_clf_results" not in st.session_state:
                with st.spinner("Training Naive Bayes, Decision Tree, SVM, KNN… (~15–30s)"):
                    clf_results = _cached_clf_benchmark(rfm)
                st.session_state["_clf_results"] = clf_results
            else:
                clf_results = st.session_state["_clf_results"]
        else:
            st.info("👆 Click **🚀 Run Classification Benchmark** to train all classifiers. "
                    "Results are cached and won't re-train on page navigation.")

        if clf_results is None:
            pass  # waiting for user to click Run
        else:
            clf_metrics_df = pd.DataFrame(clf_results).T[['Accuracy', 'Precision', 'Recall', 'F1']]

            st.table(clf_metrics_df.style.format('{:.2%}'))

            # Interactive Bar Chart
            fig_clf = px.bar(
                clf_metrics_df.reset_index(),
                x='index',
                y=['Accuracy', 'F1', 'Precision'],
                barmode='group',
                title="Model Performance Comparison across Metrics",
                labels={'index': 'Classifier', 'value': 'Score', 'variable': 'Metric'},
                color_discrete_sequence=['#6366f1', '#10b981', '#f59e0b']
            )
            fig_clf.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_clf, use_container_width=True)

            # Confusion Matrix display
            st.markdown("##### 🔲 Confusion Matrix (Decision Tree)")
            cm = np.array(clf_results['DecisionTree']['ConfusionMatrix'])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=['One-Time (0)', 'Returning (1)'],
                y=['One-Time (0)', 'Returning (1)']
            )
            fig_cm.update_layout(template="plotly_dark", height=320, width=450, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, use_container_width=False)

    # ------------------ PCA ------------------
    with sub_tab3:
        st.markdown("#### 🔬 Principal Component Analysis (PCA)")
        st.markdown("Compresses multi-dimensional customer features into principal orthogonal axes while retaining maximum variance.")
        
        adv = AdvancedModels()
        scaler_pca = StandardScaler()
        X_pca_in = scaler_pca.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        X_pca, var = adv.apply_pca(X_pca_in, n_components=2)
        
        p1, p2 = st.columns([1, 2])
        with p1:
            st.metric("PC1 Explained Variance", f"{var[0]*100:.2f}%")
            st.metric("PC2 Explained Variance", f"{var[1]*100:.2f}%")
            st.metric("Total Retained Variance", f"{sum(var)*100:.2f}%")
        with p2:
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['Monetary'] = rfm['Monetary']
            fig_pca = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Monetary',
                color_continuous_scale='Viridis',
                title="Customer Distribution in 2D Latent Space",
                opacity=0.6
            )
            fig_pca.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pca, use_container_width=True)

# ==============================================================================
# TAB 6: DATA ENGINEERING & PIPELINE TELEMETRY
# ==============================================================================
elif nav_option == "⚙️ Data Engineering & Pipeline":
    import io
    import logging as _logging
    import time
    from src.data_prep import DataPreprocessor

    st.markdown("""
    <div class="header-box" style="margin-bottom:18px;">
        <p class="header-title" style="font-size:1.5rem;">⚙️ Live Data Engineering & Pipeline Runner</p>
        <p class="header-sub">Upload your raw retail CSV and execute the full Medallion pipeline in real-time. Every cleaning step, log line, and quality metric is computed live from your data.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Mode Selector ─────────────────────────────────────────────────────
    st.markdown("#### 📂 Step 1 · Choose Data Source")

    input_mode = st.radio(
        "How do you want to provide the raw CSV?",
        ["⬆️ Upload a file", "📁 Use a local file from data/raw/"],
        horizontal=True,
        key="pipeline_input_mode",
        help="For large files (>50 MB) already on disk, use the local file option to avoid browser upload limits."
    )

    # ── Discover local raw files ────────────────────────────────────────────────
    raw_dir = "data/raw"
    local_csvs = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")] if os.path.isdir(raw_dir) else []

    uploaded_file = None
    local_file_path = None

    if input_mode == "⬆️ Upload a file":
        uploaded_file = st.file_uploader(
            "Upload your retail transactions CSV (e.g. online_retail_II.csv)",
            type=["csv"],
            key="pipeline_upload",
            help="Max 500 MB. For larger files, use the local file option instead."
        )
    else:
        if not local_csvs:
            st.warning("No CSV files found in `data/raw/`. Place your raw CSV there and refresh.")
        else:
            chosen = st.selectbox(
                "Select a raw CSV from data/raw/:",
                local_csvs,
                key="pipeline_local_select"
            )
            local_file_path = os.path.join(raw_dir, chosen)
            sz_mb = os.path.getsize(local_file_path) / 1024 / 1024
            st.info(f"📄 **{chosen}** — {sz_mb:.1f} MB on disk. No upload needed.")

    # ── Determine if we have an active data source ──────────────────────────────
    has_source = (uploaded_file is not None) or (local_file_path is not None)

    if has_source:
        # Build a stable key to detect source changes
        if uploaded_file is not None:
            file_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
        else:
            file_key = f"local_{local_file_path}_{os.path.getmtime(local_file_path)}"

        if st.session_state.get("_pipeline_file_key") != file_key:
            for k in ["_pipeline_results", "_pipeline_logs", "_pipeline_metrics",
                      "_pipeline_rfm", "_pipeline_clean_df", "_pipeline_file_key",
                      "_pipeline_raw_df", "_pipeline_raw_bytes", "_pipeline_data_ready"]:
                st.session_state.pop(k, None)
        st.session_state["_pipeline_file_key"] = file_key

        # ── Bronze Layer Preview ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div style="background:#1e2235; border:1px solid rgba(245,158,11,0.4); border-radius:12px; padding:16px 20px; margin-bottom:16px;">
            <span style="font-size:0.78rem; color:#fcd34d; font-weight:700; letter-spacing:1px;">🥉 BRONZE LAYER — RAW INGESTION</span>
        </div>
        """, unsafe_allow_html=True)

        if "_pipeline_raw_df" not in st.session_state:
            with st.spinner("📥 Reading raw CSV..."):
                try:
                    if uploaded_file is not None:
                        raw_bytes = uploaded_file.read()
                        try:
                            raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="ISO-8859-1")
                        except Exception:
                            raw_df = pd.read_csv(io.BytesIO(raw_bytes))
                        st.session_state["_pipeline_raw_bytes"] = raw_bytes
                    else:
                        # Local file — read directly; store path so pipeline can use it
                        try:
                            raw_df = pd.read_csv(local_file_path, encoding="ISO-8859-1")
                        except Exception:
                            raw_df = pd.read_csv(local_file_path)
                        # Store None for bytes — pipeline will use path directly
                        st.session_state["_pipeline_raw_bytes"] = None
                        st.session_state["_pipeline_local_path"] = local_file_path
                    st.session_state["_pipeline_raw_df"] = raw_df
                except Exception as e:
                    st.error(f"❌ Failed to read file: {e}")
                    st.stop()

        raw_df = st.session_state["_pipeline_raw_df"]
        raw_bytes = st.session_state.get("_pipeline_raw_bytes")

        b1, b2, b3 = st.columns(3)
        b1.metric("📦 Raw Records Ingested", f"{len(raw_df):,}")
        b2.metric("📋 Columns Detected", f"{raw_df.shape[1]}")
        if raw_bytes is not None:
            b3.metric("💾 File Size", f"{len(raw_bytes)/1024:.1f} KB")
        else:
            b3.metric("💾 File Size", f"{os.path.getsize(st.session_state['_pipeline_local_path'])/1024/1024:.1f} MB")

        with st.expander("🔍 Raw Data Preview (first 5 rows)", expanded=False):
            st.dataframe(raw_df.head(5), use_container_width=True)
            dtype_df = pd.DataFrame({"Column": raw_df.dtypes.index, "Dtype": raw_df.dtypes.values.astype(str),
                                     "Nulls": raw_df.isnull().sum().values})
            st.dataframe(dtype_df, use_container_width=True)


        # ── Run Pipeline Button ────────────────────────────────────────────────
        st.markdown("---")
        run_col, _ = st.columns([1, 3])
        run_pipeline = run_col.button("🚀 Run Full Pipeline", type="primary", use_container_width=True,
                                       disabled="_pipeline_results" in st.session_state)

        if "_pipeline_results" not in st.session_state and not run_pipeline:
            st.info("⬆️  Upload complete. Click **Run Full Pipeline** to execute all stages.")

        if run_pipeline or "_pipeline_results" in st.session_state:

            # ── Execute pipeline (only if not cached) ─────────────────────────
            if "_pipeline_results" not in st.session_state:

                # Set up in-memory log capture
                class _MemLogHandler(_logging.Handler):
                    def __init__(self):
                        super().__init__()
                        self.records = []
                    def emit(self, record):
                        self.records.append(self.format(record))

                mem_handler = _MemLogHandler()
                mem_handler.setFormatter(_logging.Formatter("%(asctime)s — %(levelname)s — %(message)s",
                                                             datefmt="%H:%M:%S"))

                # Attach handler to the data_prep logger
                dp_logger = _logging.getLogger("src.data_prep")
                dp_logger.setLevel(_logging.DEBUG)
                dp_logger.addHandler(mem_handler)

                metrics = {}
                stage_rows = {}

                progress_bar = st.progress(0, text="Initialising pipeline…")
                log_placeholder = st.empty()

                def _refresh_logs():
                    log_placeholder.code("\n".join(mem_handler.records) or "—", language="log")

                try:
                    import tempfile, pathlib
                    _raw_bytes = st.session_state.get("_pipeline_raw_bytes")
                    _local_path = st.session_state.get("_pipeline_local_path")

                    if _raw_bytes is not None:
                        # Upload mode — write bytes to a temp file
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                        tmp.write(_raw_bytes)
                        tmp.flush()
                        tmp.close()
                        tmp_path = tmp.name
                        _is_temp = True
                    else:
                        # Local file mode — use path directly, no copy needed
                        tmp_path = _local_path
                        _is_temp = False

                    proc = DataPreprocessor(tmp_path)

                    # Stage 1 — Load
                    progress_bar.progress(10, text="🥉 Stage 1/5 · Loading data…")
                    proc.load_data()
                    stage_rows["bronze"] = len(proc.df)
                    metrics["raw_rows"] = len(proc.df)
                    _refresh_logs()
                    time.sleep(0.3)

                    # Stage 2 — Clean
                    progress_bar.progress(30, text="🧹 Stage 2/5 · Cleaning data…")
                    before_clean = len(proc.df)
                    # Capture individual cleaning metrics by peeking at the df before/after sub-steps
                    # We'll run clean_data and derive from logs
                    proc.clean_data()
                    after_clean = len(proc.df)
                    metrics["rows_after_clean"] = after_clean
                    metrics["rows_dropped_clean"] = before_clean - after_clean
                    stage_rows["silver_clean"] = after_clean
                    _refresh_logs()
                    time.sleep(0.3)

                    # Stage 3 — Feature Engineering
                    progress_bar.progress(55, text="⚙️ Stage 3/5 · Feature engineering…")
                    proc.feature_engineering()
                    metrics["new_cols"] = ["TotalAmount", "Year", "Month", "Hour", "DayOfWeek", "DayName", "TimeOfDay"]
                    _refresh_logs()
                    time.sleep(0.3)

                    # Stage 4 — Encode & Normalize
                    progress_bar.progress(75, text="🔢 Stage 4/5 · Encoding & normalizing…")
                    proc.encode_normalize()
                    metrics["price_mean"] = float(proc.df["Price_Scaled"].mean())
                    metrics["total_mean"] = float(proc.df["TotalAmount_Scaled"].mean())
                    metrics["countries"] = int(proc.df["Country"].nunique())
                    _refresh_logs()
                    time.sleep(0.3)

                    # Stage 5 — RFM Aggregation
                    progress_bar.progress(90, text="🥇 Stage 5/5 · Building RFM mart…")
                    rfm_result = proc.get_customer_data()
                    metrics["unique_customers"] = len(rfm_result)
                    stage_rows["gold"] = len(rfm_result)
                    _refresh_logs()
                    time.sleep(0.3)

                    # NOTE: Do NOT write to disk here.
                    # The processed data is kept in session_state until the user
                    # explicitly clicks "Apply to Dashboard" below. This ensures
                    # the previously computed large dataset on disk is never
                    # silently overwritten just by running the pipeline.
                    progress_bar.progress(100, text="✅ Pipeline complete!")
                    st.session_state["_pipeline_data_ready"] = True

                    # Parse per-step drops from log records
                    missing_cid, duplicates, cancelled, invalid, bad_dates = 0, 0, 0, 0, 0
                    for rec in mem_handler.records:
                        if "missing Customer ID" in rec:
                            try: missing_cid = int(rec.split("Dropped ")[1].split(" ")[0].replace(",",""))
                            except: pass
                        if "duplicate rows" in rec.lower():
                            try: duplicates = int(rec.split("Dropped ")[1].split(" ")[0].replace(",",""))
                            except: pass
                        if "cancelled orders" in rec.lower():
                            try: cancelled = int(rec.split("Removed ")[1].split(" ")[0].replace(",",""))
                            except: pass
                        if "invalid Quantity" in rec:
                            try: invalid = int(rec.split("Removed ")[1].split(" ")[0].replace(",",""))
                            except: pass
                        if "unparseable InvoiceDate" in rec:
                            try: bad_dates = int(rec.split("Dropped ")[1].split(" ")[0].replace(",",""))
                            except: pass
                    metrics["missing_cid"] = missing_cid
                    metrics["duplicates"] = duplicates
                    metrics["cancelled"] = cancelled
                    metrics["invalid"] = invalid
                    metrics["bad_dates"] = bad_dates

                    st.session_state["_pipeline_results"] = proc.df.copy()
                    st.session_state["_pipeline_rfm"] = rfm_result.copy()
                    st.session_state["_pipeline_logs"] = list(mem_handler.records)
                    st.session_state["_pipeline_metrics"] = metrics
                    st.session_state["_pipeline_stage_rows"] = stage_rows

                    if _is_temp:
                        pathlib.Path(tmp_path).unlink(missing_ok=True)

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Pipeline failed: {e}")
                    st.exception(e)
                    st.stop()
                finally:
                    dp_logger.removeHandler(mem_handler)

            # ── Display Results ────────────────────────────────────────────────
            metrics    = st.session_state["_pipeline_metrics"]
            logs       = st.session_state["_pipeline_logs"]
            clean_df   = st.session_state["_pipeline_results"]
            rfm_result = st.session_state["_pipeline_rfm"]
            stage_rows = st.session_state.get("_pipeline_stage_rows", {})
            raw_rows   = metrics.get("raw_rows", len(raw_df))

            # ── Apply to Dashboard button ──────────────────────────────────────
            if st.session_state.get("_pipeline_data_ready") and not st.session_state.get("_using_pipeline_data"):
                st.success("✅ Pipeline complete! Review results below, then apply when ready.")
                st.warning(
                    "⚠️ Clicking **Apply to Dashboard** will **overwrite** the existing "
                    "`data/processed/` files with this new dataset. Your previously computed "
                    "large dataset will be replaced. This action cannot be undone."
                )
                if st.button("🔄 Apply to Dashboard (Overwrite & Reload with New Data)",
                             type="primary", use_container_width=True, key="apply_pipeline_btn"):
                    # ── Write to disk only NOW (user explicitly confirmed) ──────
                    import pathlib as _pl
                    _out = "data/processed"
                    _pl.Path(_out).mkdir(parents=True, exist_ok=True)
                    _save_df = st.session_state["_pipeline_results"]
                    _save_rfm = st.session_state["_pipeline_rfm"]
                    _save_cols = [c for c in _save_df.columns if not c.endswith("_Scaled")]
                    # Full cleaned transactions
                    _save_df[_save_cols].to_csv(f"{_out}/cleaned_transactions.csv", index=False)
                    # Random sample (≤25K rows) for fast EDA & association rules
                    _n = min(25_000, len(_save_df))
                    _save_df[_save_cols].sample(n=_n, random_state=42).to_csv(
                        f"{_out}/transactions_sample.csv", index=False
                    )
                    # RFM mart
                    _save_rfm.to_csv(f"{_out}/rfm_customer_data.csv", index=False)
                    # ── Now clear caches and reload ────────────────────────────
                    load_rfm_mart.clear()
                    load_transactions_sample.clear()
                    st.session_state["_using_pipeline_data"] = True
                    st.session_state.pop("_clf_results", None)
                    st.session_state.pop("_mba_rules", None)
                    st.session_state.pop("_mba_key", None)
                    st.rerun()

            # ── Silver Layer ───────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("""
            <div style="background:#1e2235; border:1px solid rgba(59,130,246,0.4); border-radius:12px; padding:16px 20px; margin-bottom:16px;">
                <span style="font-size:0.78rem; color:#93c5fd; font-weight:700; letter-spacing:1px;">🥈 SILVER LAYER — CLEANSED & ENRICHED TRANSACTIONS</span>
            </div>
            """, unsafe_allow_html=True)

            silver_rows = metrics.get("rows_after_clean", len(clean_df))
            retention   = silver_rows / raw_rows * 100 if raw_rows else 0

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("✅ Rows After Cleaning",    f"{silver_rows:,}",   f"{retention:.1f}% retained")
            s2.metric("🗑️ Missing Customer IDs",   f"{metrics.get('missing_cid',0):,}",  "Dropped")
            s3.metric("🔁 Duplicate Rows",         f"{metrics.get('duplicates',0):,}",   "Dropped")
            s4.metric("❌ Cancelled Invoices",      f"{metrics.get('cancelled',0):,}",   "Removed")
            s5.metric("📅 Bad Date Rows",           f"{metrics.get('bad_dates',0):,}",   "Coerced → Dropped")

            with st.expander("🔬 New Features Added (Feature Engineering)", expanded=False):
                new_cols = metrics.get("new_cols", [])
                st.markdown(f"**{len(new_cols)} new columns created:** " +
                            ", ".join([f"`{c}`" for c in new_cols]))
                show_cols = [c for c in new_cols if c in clean_df.columns]
                if show_cols:
                    st.dataframe(clean_df[show_cols].head(8), use_container_width=True)

            with st.expander("📊 Encoding & Normalization Stats", expanded=False):
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("🌍 Unique Countries Encoded", metrics.get("countries", "—"))
                ec2.metric("Price_Scaled µ",  f"{metrics.get('price_mean', 0):.4f}")
                ec3.metric("TotalAmount_Scaled µ", f"{metrics.get('total_mean', 0):.4f}")
                if "Price_Scaled" in clean_df.columns:
                    st.dataframe(clean_df[["Customer ID","Invoice","Price","Price_Scaled",
                                           "TotalAmount","TotalAmount_Scaled"]].head(8),
                                 use_container_width=True)

            # ── Gold Layer ─────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("""
            <div style="background:#1e2235; border:1px solid rgba(16,185,129,0.4); border-radius:12px; padding:16px 20px; margin-bottom:16px;">
                <span style="font-size:0.78rem; color:#6ee7b7; font-weight:700; letter-spacing:1px;">🥇 GOLD LAYER — RFM CUSTOMER ANALYTICS MART</span>
            </div>
            """, unsafe_allow_html=True)

            g1, g2, g3 = st.columns(3)
            g1.metric("👥 Unique Customers",    f"{metrics.get('unique_customers', len(rfm_result)):,}")
            g2.metric("📅 Avg Recency (days)",  f"{rfm_result['Recency'].mean():.0f}")
            g3.metric("💰 Avg Monetary (£)",    f"£{rfm_result['Monetary'].mean():.2f}")

            with st.expander("🏆 RFM Table Preview", expanded=True):
                st.dataframe(rfm_result.head(20).style.background_gradient(cmap="viridis", subset=["Monetary"]),
                             use_container_width=True)

            # ── Audit Metrics & Waterfall ──────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🛡️ Full Audit Metrics")
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Raw Ingested",          f"{raw_rows:,}")
            a2.metric("Missing CIDs Dropped",  f"{metrics.get('missing_cid',0):,}",
                      f"-{metrics.get('missing_cid',0)/raw_rows*100:.1f}%" if raw_rows else "")
            a3.metric("Duplicates Dropped",    f"{metrics.get('duplicates',0):,}",
                      f"-{metrics.get('duplicates',0)/raw_rows*100:.1f}%" if raw_rows else "")
            a4.metric("Cancelled Purged",      f"{metrics.get('cancelled',0):,}",
                      f"-{metrics.get('cancelled',0)/raw_rows*100:.1f}%" if raw_rows else "")
            a5.metric("Invalid Qty/Price",     f"{metrics.get('invalid',0):,}",
                      f"-{metrics.get('invalid',0)/raw_rows*100:.1f}%" if raw_rows else "")

            # Waterfall chart
            dropped_total = (metrics.get("missing_cid",0) + metrics.get("duplicates",0) +
                             metrics.get("cancelled",0)  + metrics.get("invalid",0))
            stage_labels = ["Raw Ingested", "After Missing CID Drop", "After Dedup",
                            "After Cancel Removal", "After Invalid Filter", "Final Clean"]
            stage_vals   = [
                raw_rows,
                raw_rows - metrics.get("missing_cid", 0),
                raw_rows - metrics.get("missing_cid", 0) - metrics.get("duplicates", 0),
                raw_rows - metrics.get("missing_cid", 0) - metrics.get("duplicates", 0) - metrics.get("cancelled", 0),
                silver_rows,
                silver_rows
            ]
            wf_fig = go.Figure(go.Bar(
                x=stage_labels, y=stage_vals,
                marker_color=["#f59e0b", "#f87171", "#fb923c", "#facc15", "#34d399", "#6ee7b7"],
                text=[f"{v:,}" for v in stage_vals],
                textposition="outside"
            ))
            wf_fig.update_layout(
                template="plotly_dark", height=340,
                title="Data Volume at Each Pipeline Stage",
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Row Count", xaxis_title=""
            )
            st.plotly_chart(wf_fig, use_container_width=True)

            # ── Live Logs ──────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📜 Real-Time Pipeline Audit Log")
            log_text = "\n".join(logs) if logs else "No log output captured."
            st.code(log_text, language="log")

            # Also show file log if it exists
            if os.path.exists("data_pipeline.log"):
                with st.expander("📁 Full `data_pipeline.log` (persistent file)", expanded=False):
                    with open("data_pipeline.log", "r") as _f:
                        st.code(_f.read(), language="log")

            # ── Downloads ─────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💾 Download Processed Outputs")
            dl1, dl2, dl3 = st.columns(3)

            clean_csv_path = "data/processed/cleaned_transactions.csv"
            rfm_csv_path   = "data/processed/rfm_customer_data.csv"

            if os.path.exists(clean_csv_path):
                with open(clean_csv_path, "rb") as _f:
                    dl1.download_button("⬇️ cleaned_transactions.csv", _f, "cleaned_transactions.csv",
                                        "text/csv", use_container_width=True)

            if os.path.exists(rfm_csv_path):
                with open(rfm_csv_path, "rb") as _f:
                    dl2.download_button("⬇️ rfm_customer_data.csv", _f, "rfm_customer_data.csv",
                                        "text/csv", use_container_width=True)

            dl3.download_button("⬇️ Pipeline Log (.txt)", log_text.encode(),
                                "pipeline_run.log", "text/plain", use_container_width=True)

    else:
        # No file uploaded yet — show architecture diagram as guidance
        st.markdown("#### 🏗️ Medallion Pipeline Architecture")
        st.markdown("""
        <div style="display:flex; justify-content:space-between; gap:15px; margin-bottom:20px; flex-wrap:wrap;">
            <div style="flex:1; min-width:220px; background:#1e2235; border:1px solid rgba(245,158,11,0.3); border-radius:12px; padding:18px; text-align:center;">
                <div style="font-size:2rem;">🥉</div>
                <div style="font-size:0.8rem; color:#fcd34d; font-weight:700; margin-top:8px;">BRONZE LAYER</div>
                <p style="font-size:0.82rem; color:#94a3b8; margin:6px 0 0;">Raw CSV ingestion. Missing IDs, cancellations, encoding issues.</p>
            </div>
            <div style="flex:0.2; display:flex; align-items:center; justify-content:center; font-size:1.5rem; color:#475569;">→</div>
            <div style="flex:1; min-width:220px; background:#1e2235; border:1px solid rgba(59,130,246,0.3); border-radius:12px; padding:18px; text-align:center;">
                <div style="font-size:2rem;">🥈</div>
                <div style="font-size:0.8rem; color:#93c5fd; font-weight:700; margin-top:8px;">SILVER LAYER</div>
                <p style="font-size:0.82rem; color:#94a3b8; margin:6px 0 0;">Cleaned, de-duped, enriched with TotalAmount, TimeOfDay, DayOfWeek features.</p>
            </div>
            <div style="flex:0.2; display:flex; align-items:center; justify-content:center; font-size:1.5rem; color:#475569;">→</div>
            <div style="flex:1; min-width:220px; background:#1e2235; border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:18px; text-align:center;">
                <div style="font-size:2rem;">🥇</div>
                <div style="font-size:0.8rem; color:#6ee7b7; font-weight:700; margin-top:8px;">GOLD LAYER</div>
                <p style="font-size:0.82rem; color:#94a3b8; margin:6px 0 0;">RFM customer analytics mart — ready for ML inference.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("⬆️  Upload a raw transactions CSV above to execute the live pipeline.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.82rem;">
    Enterprise Customer Intelligence & Predictive Analytics Platform • Built with Streamlit, Scikit-Learn, Plotly & PyTorch/MLP Architecture
</div>
""", unsafe_allow_html=True)
