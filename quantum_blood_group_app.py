
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pennylane as qml
from pennylane.optimize import AdamOptimizer

st.title("🧬 Quantum Blood Group Detection (QML)")
st.markdown("A hybrid quantum-classical model using PennyLane and Streamlit.")

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.choice(['A', 'B', 'AB', 'O'], 100)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Quantum device and circuit
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(x, weights):
    out = quantum_circuit(x, weights)
    return np.argmax(softmax(out))

# Training
st.write("Training model...")
num_layers = 2
weights = np.random.randn(num_layers, n_qubits)
opt = AdamOptimizer(stepsize=0.1)
batch_size = 5

for epoch in range(20):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        def cost_fn(w):
            loss = 0
            for xi, yi in zip(X_batch, y_batch):
                out = quantum_circuit(xi, w)
                pred = softmax(out)
                loss -= np.log(pred[yi])
            return loss / len(X_batch)

        weights = opt.step(cost_fn, weights)

# Predictions and evaluation
predictions = [predict(xi, weights) for xi in X_test]
acc = accuracy_score(y_test, predictions)

st.success(f"Model Accuracy: {acc:.2f}")
st.subheader("Sample Predictions")
for i in range(5):
    st.write(f"Predicted: {le.inverse_transform([predictions[i]])[0]} | Actual: {le.inverse_transform([y_test[i]])[0]}")
