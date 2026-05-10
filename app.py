"""
AutoML 
Features:
  • Forced light mode for consistent, clean contrast
  • Professional SaaS dashboard aesthetic
  • Automated preprocessing (imputation, encoding)
  • Robust RandomForest implementation (Classification/Regression)
  • Beginner & Advanced viewing modes
  • Feature importance analysis
  • Interactive inference module
  • Model export capabilities

Tech stack: streamlit · pandas · numpy · matplotlib · scikit-learn · joblib
"""

import io
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance


# PAGE CONFIGURATION


st.set_page_config(
    page_title="AutoML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# GLOBAL CSS 


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #FAFAFA;
        --bg-panel: #FFFFFF;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --border-color: #E2E8F0;
        --accent-color: #2563EB;
        --accent-hover: #1D4ED8;
        --success-color: #059669;
        --warning-color: #D97706;
        --danger-color: #DC2626;
        --radius: 8px;
    }

    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label, li {
        color: var(--text-primary) !important;
    }
    
    p, li, span {
        color: var(--text-secondary) !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    header[data-testid="stHeader"] { background: transparent !important; }
    footer { display: none !important; }

    .dashboard-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .card-header {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 12px;
    }
    .card-body {
        margin-top: 12px;
    }

    .callout {
        padding: 12px 16px;
        border-radius: var(--radius);
        margin-bottom: 16px;
        font-size: 0.9rem;
        background-color: #F8FAFC;
        border-left: 4px solid var(--border-color);
        color: var(--text-secondary) !important;
    }
    .callout-info { border-left-color: var(--accent-color); background-color: #EFF6FF; }
    .callout-success { border-left-color: var(--success-color); background-color: #ECFDF5; }
    .callout-warning { border-left-color: var(--warning-color); background-color: #FFFBEB; }

    div[data-testid="stMetric"] {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius) !important;
        padding: 16px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricValue"] {
        color: var(--accent-color) !important;
        font-weight: 700 !important;
    }

    div.stButton > button, button[kind="primary"] {
        background-color: var(--accent-color) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: var(--accent-hover) !important;
        color: #FFFFFF !important;
    }
    div.stButton > button * { color: #FFFFFF !important; }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--border-color);
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 12px;
        padding-bottom: 12px;
        color: var(--text-secondary) !important;
        font-weight: 500;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-color) !important;
        border-bottom: 2px solid var(--accent-color) !important;
    }

    .stSelectbox label, .stNumberInput label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="select"] > div, input[type="number"] {
        background-color: var(--bg-panel) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-panel) !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: var(--radius) !important;
    }

    .dataframe th {
        background-color: #F1F5F9 !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    .dataframe td {
        background-color: var(--bg-panel) !important;
        color: var(--text-secondary) !important;
    }

    .stProgress > div > div {
        background-color: var(--accent-color) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# SESSION STATE INITIALIZATION


session_defaults = {
    "model": None,
    "features": None,
    "problem_type": None,
    "is_trained": False,
    "score": None,
    "metrics": {},
    "importance_df": None,
    "advanced_mode": False,
    "cat_encoders": {},
    "target_encoder": None,
    "pipeline_logs": []
}

for key, val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# UI HELPER FUNCTIONS


def render_card(title: str, content: str = ""):
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-header">{title}</div>
            <div class="card-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_callout(text: str, type: str = "info"):
    st.markdown(f'<div class="callout callout-{type}">{text}</div>', unsafe_allow_html=True)

def handle_error(user_msg: str, tech_msg: str = ""):
    st.markdown(
        f"""
        <div class="callout callout-warning" style="border-left-color: #DC2626;">
            <strong>Task Failed:</strong> {user_msg}<br>
            <span style="font-size: 0.85rem; color: #64748B;">Detail: {tech_msg}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# SIDEBAR


with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 24px;">
            <div style="font-weight: 700; font-size: 1.25rem; color: #0F172A;">AutoML Platform</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Data processing & modeling</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Settings**")
    advanced = st.toggle(
        "Advanced Mode",
        value=st.session_state.advanced_mode,
        help="Enable technical metrics, raw data views, and advanced logs."
    )
    st.session_state.advanced_mode = advanced

    if advanced:
        render_callout("Advanced details enabled.", "info")
    else:
        render_callout("Beginner mode active. Interface simplified.", "success")

    if not advanced:
        with st.expander("Terminology Guide"):
            st.markdown(
                """
                - **Target:** The specific column you want the model to predict.
                - **Features:** The other columns used as context/clues.
                - **Classification:** Predicting categories (e.g., Status, Type).
                - **Regression:** Predicting numerical values (e.g., Price, Age).
                """
            )

    st.markdown("<div style='margin-top: 40px; font-size: 0.75rem; color: #94A3B8;'>Powered by scikit-learn</div>", unsafe_allow_html=True)


# MAIN APPLICATION FLOW

st.markdown("## Workspace")
if not st.session_state.advanced_mode:
    st.markdown("Upload a dataset to begin the automated training sequence.")
else:
    st.markdown("Configure dataset ingestion and model hyperparameters.")


uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if not uploaded_file:
    render_card(
        "System Idle",
        "Awaiting data input. Please upload a standard comma-separated values (.csv) file to initialize the workspace."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    df.columns = df.columns.str.strip()
except Exception as e:
    handle_error("Unable to parse the uploaded file.", str(e))
    st.stop()

if df.empty:
    handle_error("The provided dataset contains no records.")
    st.stop()


n_rows, n_cols = df.shape
n_missing = int(df.isnull().sum().sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{n_rows:,}")
col2.metric("Total Features", f"{n_cols:,}")
col3.metric("Missing Values", f"{n_missing:,}")

if st.session_state.advanced_mode:
    with st.expander("View Raw Dataset Head"):
        st.dataframe(df.head(), use_container_width=True)


st.markdown("### Configuration")
target_column = st.selectbox(
    "Select Target Variable",
    options=df.columns,
    help="Select the column that the model should learn to predict."
)

if target_column:
    sample_vals = df[target_column].dropna().unique()[:4]
    sample_str = ", ".join([str(v) for v in sample_vals])
    if not st.session_state.advanced_mode:
        render_callout(f"The model will learn to predict **{target_column}**. Examples of data in this column: {sample_str}")


if st.button("Initialize Training Sequence", type="primary", use_container_width=False):
    
    # Pipeline Logging
    logs = []
    
    with st.spinner("Processing data and training model..."):
        try:
            # Feature/Target Split
            X_raw = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()

            # Remove identifiers (columns where every row is unique and is string)
            id_cols = [c for c in X_raw.columns if X_raw[c].nunique() == n_rows and X_raw[c].dtype == "object"]
            if id_cols:
                X_raw = X_raw.drop(columns=id_cols)
                logs.append(f"Dropped potential identifier columns: {', '.join(id_cols)}")

            # Imputation
            num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
            cat_cols = X_raw.select_dtypes(exclude=np.number).columns.tolist()

            for col in num_cols:
                if X_raw[col].isnull().any():
                    X_raw[col] = X_raw[col].fillna(X_raw[col].median())
            
            for col in cat_cols:
                if X_raw[col].isnull().any():
                    mode_val = X_raw[col].mode()
                    X_raw[col] = X_raw[col].fillna(mode_val[0] if not mode_val.empty else "Missing")
                    
            if n_missing > 0:
                logs.append(f"Imputed {n_missing} missing values across features.")

            # Encoding Categorical Variables
            cat_encoders = {}
            cols_to_label = []
            cols_to_onehot = []

            for col in cat_cols:
                if X_raw[col].nunique() <= 10:
                    cols_to_label.append(col)
                else:
                    # Cap high cardinality
                    top_cats = X_raw[col].value_counts().head(10).index.tolist()
                    X_raw[col] = X_raw[col].where(X_raw[col].isin(top_cats), other="Other")
                    cols_to_onehot.append(col)

            for col in cols_to_label:
                le = LabelEncoder()
                X_raw[col] = le.fit_transform(X_raw[col].astype(str))
                cat_encoders[col] = le

            if cols_to_onehot:
                X_raw = pd.get_dummies(X_raw, columns=cols_to_onehot)

            X = X_raw.copy()
            X.columns = X.columns.astype(str)

            # Determine Problem Type
            if y.dtype == "object" or y.dtype == "bool" or y.nunique() <= 15:
                problem_type = "Classification"
                target_encoder = LabelEncoder() if y.dtype == "object" else None
                y_processed = pd.Series(target_encoder.fit_transform(y.astype(str))) if target_encoder else y.copy()
                logs.append("Task identified as Classification.")
            else:
                problem_type = "Regression"
                target_encoder = None
                y_processed = y.copy()
                logs.append("Task identified as Regression.")

            strat = y_processed if (problem_type == "Classification" and y_processed.nunique() > 1) else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y_processed, test_size=0.2, random_state=42, stratify=strat)
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(X, y_processed, test_size=0.2, random_state=42)

         
            if problem_type == "Classification":
                model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            logs.append(f"Trained RandomForest{problem_type} on {X_train.shape[0]} records.")

            metrics = {}
            if problem_type == "Classification":
                metrics['accuracy'] = accuracy_score(y_test, predictions)
            else:
                metrics['r2'] = r2_score(y_test, predictions)
                metrics['mae'] = mean_absolute_error(y_test, predictions)

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                perm_result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
                importances = np.clip(perm_result.importances_mean, 0, None)

            imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)

            st.session_state.model = model
            st.session_state.features = list(X.columns)
            st.session_state.problem_type = problem_type
            st.session_state.is_trained = True
            st.session_state.metrics = metrics
            st.session_state.importance_df = imp_df
            st.session_state.cat_encoders = cat_encoders
            st.session_state.target_encoder = target_encoder
            st.session_state.pipeline_logs = logs

            joblib.dump(model, "trained_model.pkl")

        except Exception as e:
            handle_error("Pipeline execution failed during training.", str(e))
            st.stop()


# POST-TRAINING DASHBOARD

if st.session_state.is_trained and st.session_state.model is not None:
    st.markdown("---")
    st.markdown("### Model Dashboard")

    tab_eval, tab_feat, tab_infer, tab_export = st.tabs([
        "Evaluation", "Feature Analysis", "Inference", "Export"
    ])


    with tab_eval:
        st.markdown("<br>", unsafe_allow_html=True)
        pt = st.session_state.problem_type
        metrics = st.session_state.metrics

        c1, c2 = st.columns([1, 2])
        with c1:
            if pt == "Classification":
                acc = metrics.get('accuracy', 0)
                st.metric("Test Accuracy", f"{acc * 100:.2f}%")
            else:
                r2 = metrics.get('r2', 0)
                mae = metrics.get('mae', 0)
                st.metric("R² Score", f"{r2:.4f}")
                st.metric("Mean Absolute Error", f"{mae:,.2f}")

        with c2:
            st.markdown("**Pipeline Execution Log**")
            for log in st.session_state.pipeline_logs:
                st.markdown(f"- {log}")
            
            if not st.session_state.advanced_mode:
                if pt == "Classification":
                    render_callout("Accuracy represents the percentage of correct predictions the model made on unseen data.", "info")
                else:
                    render_callout("R² Score indicates how well the model captures patterns in the data (1.0 is perfect).", "info")

    
    with tab_feat:
        st.markdown("<br>", unsafe_allow_html=True)
        if not st.session_state.advanced_mode:
            st.markdown("The chart below illustrates which data points were most influential in driving the model's predictions.")
        else:
            st.markdown("Relative feature importances derived from the trained RandomForest estimator.")

        imp_df = st.session_state.importance_df
        top_n = min(15, len(imp_df))
        plot_df = imp_df.head(top_n).copy()

        matplotlib.rcParams.update({
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#E2E8F0",
            "axes.labelcolor": "#475569",
            "text.color": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#F1F5F9",
        })

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
        ax.barh(
            plot_df["Feature"][::-1],
            plot_df["Importance"][::-1],
            color="#2563EB",
            height=0.5,
        )
        ax.set_xlabel("Relative Importance Score", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--")
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)

   
    with tab_infer:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("Input custom parameters to generate a live prediction.")
        
        features = st.session_state.features
        # To avoid overwhelming UI, only show top 15 features for manual input
        top_features = list(st.session_state.importance_df["Feature"].head(15))
        hidden_features = [f for f in features if f not in top_features]
        
        input_data = {}
        
        cols = st.columns(3)
        for i, feat in enumerate(top_features):
            with cols[i % 3]:
                # Simplify label formatting
                clean_label = feat.replace("_", " ").title() if not st.session_state.advanced_mode else feat
                input_data[feat] = st.number_input(clean_label, value=0.0, key=f"inf_{feat}")
        
        # Zero-fill hidden
        for feat in hidden_features:
            input_data[feat] = 0.0
            
        if st.button("Generate Prediction", type="primary"):
            try:
                input_df = pd.DataFrame([input_data])[features]
                pred = st.session_state.model.predict(input_df)[0]
                
                # Inverse transform if target was encoded
                te = st.session_state.target_encoder
                if st.session_state.problem_type == "Classification" and te is not None:
                    try:
                        pred_display = te.inverse_transform([int(pred)])[0]
                    except:
                        pred_display = pred
                else:
                    # Format regression output
                    pred_display = f"{float(pred):,.4f}" if st.session_state.advanced_mode else f"{float(pred):,.2f}"

                st.markdown(
                    f"""
                    <div style="margin-top: 16px; padding: 20px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;">
                        <span style="font-size: 0.9rem; color: #475569;">Predicted Value for {target_column}</span><br>
                        <span style="font-size: 1.75rem; font-weight: 700; color: #0F172A;">{pred_display}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                handle_error("Inference generation failed.", str(e))


    with tab_export:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("Download the serialized model artifact for deployment.")
        
        buf = io.BytesIO()
        try:
            joblib.dump(st.session_state.model, buf)
            buf.seek(0)
            
            st.download_button(
                label="Download model.pkl",
                data=buf,
                file_name="model.pkl",
                mime="application/octet-stream",
                type="primary"
            )
            
            if st.session_state.advanced_mode:
                st.markdown("### Metadata")
                st.code(
                    f"Algorithm: {type(st.session_state.model).__name__}\n"
                    f"Task Type: {st.session_state.problem_type}\n"
                    f"Features Count: {len(st.session_state.features)}\n"
                    f"scikit-learn Version: {joblib.__version__}",
                    language="yaml"
                )
        except Exception as e:
            handle_error("Serialization failed.", str(e))