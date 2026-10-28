"""A-44 (check): re-run the scoring of A-39 to A-43 on the raw curve files and compare with the saved score files."""
import subprocess, sys
def run(a): return subprocess.run([sys.executable] + a, capture_output=True, text=True).stdout.strip()
for args, saved, n in [(["a31_score.py", "a39", "2500"], "a39_scores.txt", 2), (["a40_score.py"], "a40_scores.txt", None), (["a41b_score.py"], "a41b_scores.txt", None), (["a31_score.py", "a43", "2500"], "a43_scores.txt", 2)]:
    want = open(saved).read().strip().splitlines(); got = run(args).splitlines()
    k = len(got) if n is None else n
    print(f"{' '.join(args)} -> matches first {k} lines of {saved}: {got[:k] == want[:k]}")
