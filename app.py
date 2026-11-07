import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    confusion_matrix, classification_report

st.set_page_config(page_title="Adult Income ML App", layout="wide")
st.title("Income Classification Models App")
st.markdown("Upload the `test_data.csv` file generated during training to evaluate the models.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")

# 1. Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    # Reading the data
    df = pd.read_csv(uploaded_file)
    st.write("**Dataset Preview:**", df.head())

    # Separate features and target
    X = df.drop(columns=['income_target'])
    y_true = df['income_target']

    # 2. Model Selection
    model_name = st.sidebar.selectbox(
        "Select Machine Learning Model",
        ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
    )

    # Map selection to filename
    file_map = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'KNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Support Vector Machine': 'support_vector_machine.pkl'
    }

    # Load model ####
    model = joblib.load(f"model/{file_map[model_name]}")

    # Predictions
    y_pred = model.predict(X)
    # Get probabilities for AUC if supported
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # 3. Display Metrics
    st.subheader("Evaluation Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
    col2.metric("Precision", round(precision_score(y_true, y_pred), 4))
    col3.metric("Recall", round(recall_score(y_true, y_pred), 4))
    col4.metric("F1 Score", round(f1_score(y_true, y_pred), 4))
    col5.metric("MCC Score", round(matthews_corrcoef(y_true, y_pred), 4))

    if y_prob is not None:
        col6.metric("AUC Score", round(roc_auc_score(y_true, y_prob), 4))
    else:
        col6.metric("AUC Score", "N/A")

    # 4. Confusion Matrix & Classification Report
    st.subheader("Model Performance Analysis")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

        with colB:
            st.markdown("**Classification Report**")
            report_dict = classification_report(y_true, y_pred, target_names=['<=50K', '>50K'], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=3))
else:
    st.info("Awaiting CSV file upload in the sidebar to display metrics.")