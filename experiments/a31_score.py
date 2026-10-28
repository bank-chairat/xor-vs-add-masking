import csv, glob, sys, numpy as np
for f in sorted(glob.glob(sys.argv[1] + '/*_curves.csv')):
    R = list(csv.DictReader(open(f))); tag = f.split('/')[-1].replace('_curves.csv', '')
    out = []
    for s in sorted(set(r['run_id'] for r in R)):
        rr = sorted([r for r in R if r['run_id'] == s], key=lambda r: int(r['step'])); acol = [k for k in rr[0] if 'A' in k and 'val' in k][0]
        fs = next((int(r['step']) for r in rr if int(r['step']) >= 100 and int(r['step']) % 100 == 0 and float(r[acol]) < 0.9 * 0.0353), -1)
        out.append(fs)
    print(tag, 'flat start per seed', out, 'median', np.median([x if x > 0 else int(sys.argv[2]) for x in out]))
print('references: c = 1 (A-15/A-28): 900 1300 1300 1200 (median 1,250); c = 2 (all layers): no escape by 2,500 in 4 of 4')
