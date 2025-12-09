import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.datasets import make_classification

# --- Helper Function for Model Training and Evaluation ---
def evaluate_model(majority_ratio):
    """Generates imbalanced data, trains a model, and returns metrics."""

    # 1. Generate Imbalanced Dataset
    N_samples = 1000
    # Calculate the number of minority samples based on the ratio (e.g., 9:1 ratio means 100 minority, 900 majority)
    N_minority = int(N_samples / (1 + majority_ratio))
    N_majority = N_samples - N_minority

    # Define weights for make_classification: [Class 0 (Majority), Class 1 (Minority)]
    weights = [N_majority / N_samples, N_minority / N_samples] 

    # Use make_classification to generate a dataset
    X, y = make_classification(
        n_samples=N_samples,
        n_features=5, # Use a few features
        n_redundant=0,
        n_informative=3,
        n_clusters_per_class=1,
        weights=weights,
        flip_y=0,
        random_state=42
    )

    # 2. Split Data and Train Model
    # stratify=y ensures the imbalance ratio is preserved in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Use a simple Logistic Regression
    # We use class_weight=None (default) to show the model's natural bias toward the majority class
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 3. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    # F1-Score is calculated for the MINORITY class (1) by default
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, f1, cm, y_test.sum(), len(y_test) - y_test.sum()

# --- Streamlit App Layout ---
def main():
    st.title('Accuracy vs. F1-Score for Imbalanced Data')

    st.markdown("""
    This app demonstrates how **Accuracy** can be misleading for **imbalanced datasets**. 
    Use the slider to increase the imbalance and observe the drop in the **F1-Score**.
    """)

    st.sidebar.header('Configuration')
    
    # Slider for user input
    majority_ratio = st.sidebar.slider(
        'Majority Class to Minority Class Ratio (Class 0 : Class 1)',
        min_value=1.0, # Balanced
        max_value=10.0, # Highly Imbalanced
        value=8.0,
        step=0.5,
        format='%.1f : 1'
    )
    
    # Execute the evaluation function
    accuracy, f1, cm, minority_count, majority_count = evaluate_model(majority_ratio)
    
    # --- Results Display ---
    st.header('Dataset and Results')

    st.markdown(f"""
    * **Imbalance Ratio:** **{majority_ratio:.1f} : 1**
    * **Total Test Samples:** 300
    * **Majority Class (0) Count:** {majority_count}
    * **Minority Class (1) Count:** {minority_count}
    """)
    
    st.subheader('Key Metric Comparison')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Overall Accuracy", value=f"{accuracy:.4f}")
        st.markdown("""
        Accuracy is the proportion of **total** correct predictions.
        With a high ratio, it's inflated by correct predictions of the **Majority Class**.
        """)

    with col2:
        st.metric(label="F1-Score (Minority Class 1)", value=f"{f1:.4f}")
        st.markdown("""
        **F1-Score** balances **Precision** and **Recall**. 
        It reveals the model's true failure to capture the rare **Minority Class**.
        """)

    st.markdown("---")

    st.subheader('Confusion Matrix: Where Accuracy Fails')
    
    # Display the Confusion Matrix in a nice table
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Class 0 (Majority)', 'Actual Class 1 (Minority)'],
        columns=['Predicted Class 0', 'Predicted Class 1']
    )
    st.table(cm_df)
    
    st.markdown("""
    

[Image of Confusion Matrix labeled with TP, TN, FP, FN]

    The matrix shows the problem: When the ratio is high, the model achieves high $\text{Accuracy}$ because of the high number of **True Negatives ($\text{TN}$)**. However, the high number of **False Negatives ($\text{FN}$)** (missing the minority class) pulls the $\text{F1}$-Score down.
    """)
    
    st.markdown("""
    $$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
    """)

if __name__ == '__main__':
    main()