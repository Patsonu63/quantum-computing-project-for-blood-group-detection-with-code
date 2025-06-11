
# Blood Group Detection using Quantum Machine Learning (Hybrid Model)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pennylane as qml
from pennylane.optimize import AdamOptimizer

# Step 1: Dataset Preparation
np.random.seed(42)
X = np.random.rand(100, 2)  # Simulated features
y = np.random.choice(['A', 'B', 'AB', 'O'], 100)

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Step 2: Define Quantum Device and Circuit
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Step 3: Prediction and Training
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(x, weights):
    out = quantum_circuit(x, weights)
    return np.argmax(softmax(out))

# Initialize weights and optimizer
num_layers = 2
weights = np.random.randn(num_layers, n_qubits)
opt = AdamOptimizer(stepsize=0.1)
batch_size = 5

# Training Loop
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
    print(f"Epoch {epoch + 1}: Cost = {cost_fn(weights):.4f}")

# Evaluation
predictions = [predict(xi, weights) for xi in X_test]
acc = accuracy_score(y_test, predictions)
print("Accuracy:", acc)
print("Predicted:", le.inverse_transform(predictions[:10]))
print("Actual:   ", le.inverse_transform(y_test[:10]))
