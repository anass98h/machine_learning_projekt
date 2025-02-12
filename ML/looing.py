import subprocess

# Full path to your classifier_training.py script
MAIN_SCRIPT = r"c:/Users/anass/Desktop/ml project/machine_learning_projekt/ML/KNN.py"

# Path to Python 3.13
PYTHON_EXE = r"C:/Users/anass/AppData/Local/Microsoft/WindowsApps/python3.13.exe"

for n in range(1, 39):
    print(f"=== Running experiment with n_neighbors={n} ===")
    subprocess.run([PYTHON_EXE, MAIN_SCRIPT, str(n)], check=True)
