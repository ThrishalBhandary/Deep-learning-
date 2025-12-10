import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Set up the Streamlit page ---
st.set_page_config(
    page_title="Vanishing/Exploding Gradient Demo",
    layout="wide"
)

st.title("📉 Vanishing/Exploding Gradient Demo")

st.markdown("""
This application simulates the core mechanism behind the **Vanishing** and **Exploding Gradient** problems in deep sequential models (like an unrolled RNN or a very deep Feedforward Network).

The magnitude of the gradient after $T$ steps (sequence length) is proportional to the product of $T$ derivatives/Jacobians from each step: $\\text{Gradient} \\propto \\prod_{t=1}^{T} \\frac{\\partial h_t}{\\partial h_{t-1}}$.
Here, we model the derivative at each step, $\\frac{\\partial h_t}{\\partial h_{t-1}}$, as a single value (Weight Magnitude).
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")

    # Sequence Length (T)
    sequence_length = st.slider(
        "Sequence Length ($T$):",
        min_value=2,
        max_value=100,
        value=50,
        step=1,
        help="Simulates the number of layers (depth) or time steps in a sequence."
    )

    # Weight Magnitude (w)
    weight_magnitude = st.slider(
        "Weight Magnitude ($|w|$):",
        min_value=0.1,
        max_value=3.0,
        value=1.5,
        step=0.05,
        format="%.2f",
        help="Simulates the average magnitude of the recurrent weight matrix's eigenvalues/derivatives."
    )

    # Initial Gradient
    initial_gradient = st.slider(
        "Initial Gradient Value:",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="The gradient value at the last time step/layer."
    )

    st.markdown("---")
    st.subheader("💡 Observations")
    st.markdown("""
    * **Vanishing Gradient:** Occurs when $|w| < 1$. The gradient shrinks exponentially to zero.
    * **Exploding Gradient:** Occurs when $|w| > 1$. The gradient grows exponentially to a very large number.
    * **Stable Gradient:** Occurs when $|w| \approx 1$. The gradient remains stable.
    """)

# --- Simulation Logic ---

# Create an array of time steps (t = T, T-1, ..., 1)
# We are backpropagating, so we go from T down to 1
steps = np.arange(sequence_length, 0, -1)
gradients = []
current_gradient = initial_gradient

# Simulate the backpropagation process
for step in range(sequence_length):
    # In backprop, the gradient at layer t-1 is the gradient at layer t multiplied by the derivative at t
    # For simplicity, we use the weight magnitude as the derivative/Jacobian factor
    current_gradient *= weight_magnitude
    gradients.append(current_gradient)

# Reverse the array so the x-axis (steps) goes from 1 to T (from input layer to output layer)
gradients = np.array(gradients)[::-1]
data = pd.DataFrame({
    'Time Step / Layer': np.arange(1, sequence_length + 1),
    'Gradient Magnitude': gradients
})

# --- Display Results ---

# 1. Gradient Value at First Step
final_gradient_val = gradients[0]
gradient_status = "Stable"
color = "green"

if weight_magnitude < 1:
    gradient_status = "Vanishing"
    color = "red"
elif weight_magnitude > 1.05: # Use a slight buffer for 'exploding' clarity
    gradient_status = "Exploding"
    color = "blue"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Simulation Results")
    st.metric(
        label=f"Gradient at Time Step/Layer 1 (Input End) is **{gradient_status}**",
        value=f"{final_gradient_val:.2e}",
        delta=f"Based on $|w| = {weight_magnitude:.2f}$"
    )

    # Display an explanation based on the status
    if gradient_status == "Vanishing":
        st.error(f"**Vanishing Gradient:** The gradient has shrunk to {final_gradient_val:.2e}. The early layers/steps will learn very little or stop learning entirely.")
    elif gradient_status == "Exploding":
        st.warning(f"**Exploding Gradient:** The gradient has grown to {final_gradient_val:.2e}. This can lead to massive weight updates, causing the model to become unstable, outputting `NaN`s, or 'Diverge'.")
    else:
        st.success("**Stable Gradient:** The gradient remains manageable across the sequence length, allowing effective learning at all layers/steps.")

with col2:
    st.subheader("Gradient Calculation")
    st.latex(f"""
    \\text{{Gradient}}_1 \\approx \\text{{Initial Gradient}} \\times (|w|)^{sequence_length}
    """)
    st.markdown(f"""
    This is approximately:
    $${initial_gradient} \\times ({weight_magnitude:.2f})^{sequence_length} \\approx {final_gradient_val:.2e}$$
    """)

# 2. Plotting the Gradient Flow
st.subheader("Gradient Magnitude Across Sequence Steps")
#  (This visual helps ground the "steps" context)
# The image tag is intended to show the unrolled RNN structure to clarify the "sequence steps" or "layers"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Time Step / Layer'], data['Gradient Magnitude'], marker='o', linestyle='-', color=color, markersize=4)

if gradient_status == "Exploding":
    # Use a log scale for the y-axis to visualize the exponential growth clearly
    ax.set_yscale('log')
    ax.set_ylabel("Gradient Magnitude (Log Scale)")
else:
    # Use a linear scale for a clearer view of vanishing/stable
    ax.set_ylabel("Gradient Magnitude")

ax.set_xlabel(f"Time Step / Layer (From Input End, Sequence Length = {sequence_length})")
ax.set_title(f"Gradient Flow during Backpropagation (Weight Magnitude $|w|={weight_magnitude:.2f}$)")
ax.grid(True, which="both", ls="--")
st.pyplot(fig)