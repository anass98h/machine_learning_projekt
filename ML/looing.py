import re
import ast
import pandas as pd
import matplotlib.pyplot as plt

# Multiline string containing the new output.
output_text = """
CV Accuracy: 0.8156, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'linear'}
CV Accuracy: 0.1650, Std: 0.0281, Params: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.1422, Std: 0.0279, Params: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'poly'}
CV Accuracy: 0.1648, Std: 0.0282, Params: {'clf__C': 0.1, 'clf__gamma': 0.0001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8156, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'linear'}
CV Accuracy: 0.5317, Std: 0.0232, Params: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.1422, Std: 0.0279, Params: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'poly'}
CV Accuracy: 0.3668, Std: 0.0208, Params: {'clf__C': 0.1, 'clf__gamma': 0.001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8156, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'linear'}
CV Accuracy: 0.7071, Std: 0.0116, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}
CV Accuracy: 0.2802, Std: 0.0320, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'poly'}
CV Accuracy: 0.6473, Std: 0.0080, Params: {'clf__C': 0.1, 'clf__gamma': 0.01, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8156, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'linear'}
CV Accuracy: 0.8546, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9586, Std: 0.0046, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'poly'}
CV Accuracy: 0.2788, Std: 0.0257, Params: {'clf__C': 0.1, 'clf__gamma': 0.1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8156, Std: 0.0095, Params: {'clf__C': 0.1, 'clf__gamma': 1, 'clf__kernel': 'linear'}
CV Accuracy: 0.7530, Std: 0.0607, Params: {'clf__C': 0.1, 'clf__gamma': 1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 0.1, 'clf__gamma': 1, 'clf__kernel': 'poly'}
CV Accuracy: 0.0675, Std: 0.0272, Params: {'clf__C': 0.1, 'clf__gamma': 1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8498, Std: 0.0022, Params: {'clf__C': 1, 'clf__gamma': 0.0001, 'clf__kernel': 'linear'}
CV Accuracy: 0.5319, Std: 0.0210, Params: {'clf__C': 1, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.1422, Std: 0.0279, Params: {'clf__C': 1, 'clf__gamma': 0.0001, 'clf__kernel': 'poly'}
CV Accuracy: 0.3668, Std: 0.0208, Params: {'clf__C': 1, 'clf__gamma': 0.0001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8498, Std: 0.0022, Params: {'clf__C': 1, 'clf__gamma': 0.001, 'clf__kernel': 'linear'}
CV Accuracy: 0.6976, Std: 0.0082, Params: {'clf__C': 1, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.1422, Std: 0.0279, Params: {'clf__C': 1, 'clf__gamma': 0.001, 'clf__kernel': 'poly'}
CV Accuracy: 0.6567, Std: 0.0094, Params: {'clf__C': 1, 'clf__gamma': 0.001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8498, Std: 0.0022, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'linear'}
CV Accuracy: 0.8586, Std: 0.0051, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}
CV Accuracy: 0.7187, Std: 0.0066, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'poly'}
CV Accuracy: 0.7279, Std: 0.0112, Params: {'clf__C': 1, 'clf__gamma': 0.01, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8498, Std: 0.0022, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'linear'}
CV Accuracy: 0.9627, Std: 0.0069, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'poly'}
CV Accuracy: 0.2802, Std: 0.0270, Params: {'clf__C': 1, 'clf__gamma': 0.1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8498, Std: 0.0022, Params: {'clf__C': 1, 'clf__gamma': 1, 'clf__kernel': 'linear'}
CV Accuracy: 0.9503, Std: 0.0077, Params: {'clf__C': 1, 'clf__gamma': 1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 1, 'clf__gamma': 1, 'clf__kernel': 'poly'}
CV Accuracy: 0.0724, Std: 0.0273, Params: {'clf__C': 1, 'clf__gamma': 1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8614, Std: 0.0035, Params: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'linear'}
CV Accuracy: 0.6960, Std: 0.0063, Params: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.1422, Std: 0.0279, Params: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'poly'}
CV Accuracy: 0.6567, Std: 0.0094, Params: {'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8614, Std: 0.0035, Params: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'linear'}
CV Accuracy: 0.7891, Std: 0.0055, Params: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.1422, Std: 0.0279, Params: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'poly'}
CV Accuracy: 0.7497, Std: 0.0130, Params: {'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8614, Std: 0.0035, Params: {'clf__C': 10, 'clf__gamma': 0.01, 'clf__kernel': 'linear'}
CV Accuracy: 0.9476, Std: 0.0059, Params: {'clf__C': 10, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9219, Std: 0.0085, Params: {'clf__C': 10, 'clf__gamma': 0.01, 'clf__kernel': 'poly'}
CV Accuracy: 0.7306, Std: 0.0105, Params: {'clf__C': 10, 'clf__gamma': 0.01, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8614, Std: 0.0035, Params: {'clf__C': 10, 'clf__gamma': 0.1, 'clf__kernel': 'linear'}
CV Accuracy: 0.9637, Std: 0.0058, Params: {'clf__C': 10, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 10, 'clf__gamma': 0.1, 'clf__kernel': 'poly'}
CV Accuracy: 0.3041, Std: 0.0342, Params: {'clf__C': 10, 'clf__gamma': 0.1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8614, Std: 0.0035, Params: {'clf__C': 10, 'clf__gamma': 1, 'clf__kernel': 'linear'}
CV Accuracy: 0.9503, Std: 0.0077, Params: {'clf__C': 10, 'clf__gamma': 1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 10, 'clf__gamma': 1, 'clf__kernel': 'poly'}
CV Accuracy: 0.0792, Std: 0.0264, Params: {'clf__C': 10, 'clf__gamma': 1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8702, Std: 0.0043, Params: {'clf__C': 300, 'clf__gamma': 0.0001, 'clf__kernel': 'linear'}
CV Accuracy: 0.8135, Std: 0.0075, Params: {'clf__C': 300, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.1422, Std: 0.0279, Params: {'clf__C': 300, 'clf__gamma': 0.0001, 'clf__kernel': 'poly'}
CV Accuracy: 0.7861, Std: 0.0042, Params: {'clf__C': 300, 'clf__gamma': 0.0001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8702, Std: 0.0043, Params: {'clf__C': 300, 'clf__gamma': 0.001, 'clf__kernel': 'linear'}
CV Accuracy: 0.9232, Std: 0.0037, Params: {'clf__C': 300, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}
CV Accuracy: 0.4650, Std: 0.0121, Params: {'clf__C': 300, 'clf__gamma': 0.001, 'clf__kernel': 'poly'}
CV Accuracy: 0.8346, Std: 0.0070, Params: {'clf__C': 300, 'clf__gamma': 0.001, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8702, Std: 0.0043, Params: {'clf__C': 300, 'clf__gamma': 0.01, 'clf__kernel': 'linear'}
CV Accuracy: 0.9592, Std: 0.0042, Params: {'clf__C': 300, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0055, Params: {'clf__C': 300, 'clf__gamma': 0.01, 'clf__kernel': 'poly'}
CV Accuracy: 0.7421, Std: 0.0100, Params: {'clf__C': 300, 'clf__gamma': 0.01, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8702, Std: 0.0043, Params: {'clf__C': 300, 'clf__gamma': 0.1, 'clf__kernel': 'linear'}
CV Accuracy: 0.9637, Std: 0.0058, Params: {'clf__C': 300, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 300, 'clf__gamma': 0.1, 'clf__kernel': 'poly'}
CV Accuracy: 0.3196, Std: 0.0368, Params: {'clf__C': 300, 'clf__gamma': 0.1, 'clf__kernel': 'sigmoid'}
CV Accuracy: 0.8702, Std: 0.0043, Params: {'clf__C': 300, 'clf__gamma': 1, 'clf__kernel': 'linear'}
CV Accuracy: 0.9503, Std: 0.0077, Params: {'clf__C': 300, 'clf__gamma': 1, 'clf__kernel': 'rbf'}
CV Accuracy: 0.9589, Std: 0.0046, Params: {'clf__C': 300, 'clf__gamma': 1, 'clf__kernel': 'poly'}
CV Accuracy: 0.0859, Std: 0.0270, Params: {'clf__C': 300, 'clf__gamma': 1, 'clf__kernel': 'sigmoid'}


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
