import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--file_pattern', type=str, default="ft_%d.npy")
parser.add_argument('--dmp_file_pattern', type=str, default=None)
parser.add_argument('--eps', type=int, nargs='+', default=[-1])
parser.add_argument('--clip', type=float, default=np.inf)
parser.add_argument('--use_ep_range', action='store_true')  # excludes the end
parser.add_argument('--legend', action='store_true')
parser.add_argument('--allow_missing', action='store_true')
parser.add_argument('--forces_only', action='store_true',
                    help="Use forces only (otherwise uses all forces and torques")


args = parser.parse_args()

if len(args.eps) == 1:
    if args.eps[0] == -1:
        # all
        raise NotImplementedError
    else:
        # 0 ... max
        eps = list(range(args.eps[0]))

else:
    eps = args.eps
    if args.use_ep_range:
        assert len(eps) == 2
        eps = list(range(eps[0], eps[1]))

# file path
path = os.path.abspath(args.file_pattern)
pp = os.path.dirname(path)
assert os.path.exists(pp), pp

if args.dmp_file_pattern is not None:
    dmp_path = os.path.abspath(args.dmp_file_pattern)
    dp = os.path.dirname(path)
    assert os.path.exists(dp), dp
else:
    dmp_path = None

all_ft_seqs = []
all_dmp_seqs = {}

for e in eps:
    # pattern must contain this
    fname = path % e
    if args.allow_missing and not os.path.exists(fname):
        continue
    else:
        force_torques = np.load(fname)
        all_ft_seqs.append(np.clip(force_torques, -args.clip, args.clip))

    if dmp_path is not None:
        fname = dmp_path % e
        if args.allow_missing and not os.path.exists(fname):
            continue
        else:
            dmp_all = np.load(fname)
            for key in dmp_all.keys():
                if key not in all_dmp_seqs.keys():
                    all_dmp_seqs[key] = []
                all_dmp_seqs[key].append(np.clip(dmp_all[key], -args.clip, args.clip))

# E lists (Ni, 3), for each
forces = [fseq[..., :3] for fseq in all_ft_seqs]
torques = [fseq[..., 3:] for fseq in all_ft_seqs]

if dmp_path is not None:
    desired_forces = [dseq[:, :3] for dseq in all_dmp_seqs['haptic_target']]

fig = plt.figure(figsize=(10,10), tight_layout=True)
axes = fig.subplots(nrows=3, ncols=3)

labels = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz', 'xdd', 'ydd', 'zdd']
max_x = 1
for seq in all_ft_seqs:
    max_x = max(max_x, seq.shape[0])

episode_colors = plt.get_cmap('tab20', len(eps))

for r in range(3):
    for c in range(3):
        ax = axes[r][c]
        ax.set_title(labels[r * 3 + c])
        if r == 0:
            for i, seq in zip(range(len(eps)), forces):
                ln = min(max_x, seq.shape[0])
                if dmp_path is not None:
                    htarg = desired_forces[i]
                    ln = min(ln, htarg.shape[0])
                    ax.plot(range(ln), htarg[:ln, c], c=episode_colors(i), linestyle='dashed')
                ax.plot(range(ln), seq[:ln, c], c=episode_colors(i))
        elif r < 2:
            for i, seq in enumerate(all_ft_seqs):
                # print(seq.shape)
                ax.plot(range(seq.shape[0]), seq[:, r * 3 + c], c=episode_colors(i))
        else:
            if dmp_path is not None:
                added = {}
                for key, ls_vals in all_dmp_seqs.items():
                    for i, seq in enumerate(ls_vals):
                        ln = min(max_x, seq.shape[0])
                        if key in added.keys():
                            ax.plot(range(ln), seq[:ln, c], label='_', c=added[key])
                        else:
                            # next color
                            added[key] = plt.get_cmap('Accent', len(all_dmp_seqs.keys()))(len(added.keys()))
                            ax.plot(range(ln), seq[:ln, c], label=key, c=added[key])
                ax.legend()

plt.show()