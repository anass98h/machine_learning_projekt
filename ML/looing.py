import re
import ast
import pandas as pd
import matplotlib.pyplot as plt

# Multiline string containing the new output.
output_text = """
CV Accuracy: 0.8156, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'linear'}
CV Accuracy: 0.8546, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}     
CV Accuracy: 0.9586, Std: 0.0046, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'poly'}    
CV Accuracy: 0.2788, Std: 0.0257, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'sigmoid'} 
CV Accuracy: 0.8156, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'linear'} 
CV Accuracy: 0.7071, Std: 0.0116, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}    
CV Accuracy: 0.2802, Std: 0.0320, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'poly'}   
CV Accuracy: 0.6473, Std: 0.0080, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8498, Std: 0.0022, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'linear'}    
CV Accuracy: 0.9627, Std: 0.0069, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}       
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'poly'}      
CV Accuracy: 0.2802, Std: 0.0270, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8498, Std: 0.0022, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'linear'}
CV Accuracy: 0.8586, Std: 0.0051, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}
CV Accuracy: 0.7187, Std: 0.0066, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'poly'}
CV Accuracy: 0.7279, Std: 0.0112, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'sigmoid'}
"""

# Parse the output text into a list of records.
lines = output_text.strip().splitlines()
records = []
pattern = r"CV Accuracy:\s*([\d.]+),\s*Std:\s*([\d.]+),\s*Params:\s*(\{.*\})"
for line in lines:
    match = re.match(pattern, line)
    if match:
        mean_acc = float(match.group(1))
        std_acc = float(match.group(2))
        params_str = match.group(3)
        params = ast.literal_eval(params_str)
        records.append({
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "C": params['clf__C'],
            "gamma": params['clf__gamma'],
            "kernel": params['clf__kernel']
        })

# Create a DataFrame.
df = pd.DataFrame(records)
print("Parsed DataFrame:")
print(df.head())

# Identify the best parameter combination (highest CV accuracy).
best_index = df['mean_accuracy'].idxmax()
best_entry = df.loc[best_index]
print("\nBest Parameter Combination:")
print(best_entry)

# ----------------------------------------------------
# Create separate figures for each kernel (Errorbar plots)
# ----------------------------------------------------
kernels = df['kernel'].unique()
gammas = sorted(df['gamma'].unique())

for kernel in kernels:
    subset = df[df['kernel'] == kernel]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('log')
    for gamma in gammas:
        gamma_subset = subset[subset['gamma'] == gamma].sort_values(by='C')
        if not gamma_subset.empty:
            ax.errorbar(gamma_subset['C'],
                        gamma_subset['mean_accuracy'],
                        yerr=gamma_subset['std_accuracy'],
                        marker='o', linestyle='-', label=f'gamma: {gamma}')
    ax.set_title(f"Kernel: {kernel}")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("CV Accuracy")
    ax.grid(True)
    # Highlight the best parameter combination if it belongs to this kernel.
    if kernel == best_entry['kernel']:
        ax.plot(best_entry['C'], best_entry['mean_accuracy'],
                marker='s', markersize=12, color='red', label='Best Score')
        ax.annotate("Best", (best_entry['C'], best_entry['mean_accuracy']),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    ax.legend()
    plt.tight_layout()
    plt.show()  # Opens a new window for each kernel plot.
