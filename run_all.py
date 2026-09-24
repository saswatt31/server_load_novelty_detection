"""Run the full server-load novelty-detection pipeline in one go."""
import subprocess
import sys

STEPS = [
    "generate_data.py",
    "detect_anomalies.py",
    "evaluate.py",
    "plot_results.py",
    "compare_baselines.py",
]

for script in STEPS:
    print(f"\n=== {script} " + "=" * 50)
    ret = subprocess.run([sys.executable, script]).returncode
    if ret != 0:
        sys.exit(f"{script} failed with exit code {ret}")
print("\nAll done. See results/ for outputs.")
print("Report: python make_report.py && python check_report_template.py")
