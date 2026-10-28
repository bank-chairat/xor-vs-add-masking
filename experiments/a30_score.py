import csv, numpy as np
R = list(csv.DictReader(open('a30/A-30_curves.csv'))); F = list(csv.DictReader(open('a30/A-30_final.csv')))
def flat_start(rows):
    rows = sorted(rows, key=lambda r: int(r['step'])); a25 = [float(r['val_A_parity_group']) for r in rows if int(r['step']) == 25][0]
    for r in rows:
        if int(r['step']) >= 100 and int(r['step']) % 100 == 0 and float(r['val_A_parity_group']) < 0.9 * a25: return int(r['step'])
    return -1
for K in sorted(set(r['K'] for r in R)):
    fs = []; As = []
    for s in sorted(set(r['seed'] for r in R if r['K'] == K)):
        fs.append(flat_start([r for r in R if r['K'] == K and r['seed'] == s])); As.append(float([f for f in F if f['K'] == K and f['seed'] == s][0]['test_A']))
    print(f'K {K}: flat start per seed {fs} median {np.median([x if x > 0 else 4000 for x in fs]):.0f} | A at 4,000 {np.round(As, 4).tolist()} mean {np.mean(As):.4f}')
print('references (A-24): hard labels flat start 1100 1200 900 1700 (median 1150), A 0.0105; soft labels 100 in all seeds, A 0.0004')
