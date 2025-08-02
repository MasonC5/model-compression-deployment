# Benchmark Visualization (benchmark_visualization.ipynb)
import matplotlib.pyplot as plt
import pandas as pd

# Manually enter the results from the Python script
results = {
    'Model': ['Baseline', 'Quantized', 'Pruned', 'Distilled'],
    'Accuracy (%)': [85.0, 83.0, 82.0, 84.5],  # Replace with actual values
    'Inference Time (s)': [5.2, 3.4, 4.8, 4.1],  # Replace with actual values
    'Size (MB)': [5.6, 1.3, 3.2, 1.8]  # Replace with actual values
}

# Create DataFrame
df = pd.DataFrame(results)

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.bar(df['Model'], df['Accuracy (%)'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot Inference Time
plt.figure(figsize=(8, 5))
plt.bar(df['Model'], df['Inference Time (s)'])
plt.title('Model Inference Time Comparison')
plt.ylabel('Time (s)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot Model Size
plt.figure(figsize=(8, 5))
plt.bar(df['Model'], df['Size (MB)'])
plt.title('Model Size Comparison')
plt.ylabel('Size (MB)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
