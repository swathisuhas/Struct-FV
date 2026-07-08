import re

inp = "/BS/feature_viz/work/code/GraphMatchingFV/class_fvs/missing_runs.txt"
out = "missing_runs_by_class.txt"

with open(inp, "r") as f, open(out, "w") as g:
    for line in f:
        m = re.match(r"^\s*(\d+)\s*:\s*\[(.*?)\]\s*$", line)
        if not m:
            continue
        cls = m.group(1)
        runs = m.group(2).replace(",", "")
        g.write(f"{cls} {runs}\n")