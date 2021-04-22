import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--file_pattern', type=str, default="%d/ft_%d.npy")
parser.add_argument('--eps', type=int, nargs='+', default=[-1])
parser.add_argument('--clip', type=float, default=np.inf)
parser.add_argument('--tasks', type=int, nargs='+', default=[-1])
parser.add_argument('--labels', type=str, nargs='*', default=[])
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

assert len(args.tasks) == len(np.unique(args.tasks)), args.tasks
tasks = args.tasks
labels = [str(s) for s in tasks] if len(args.labels) == 0 else args.labels
assert len(labels) == len(tasks)

# file path
path = os.path.abspath(args.file_pattern)
# pp = os.path.dirname(path)
# assert os.path.exists(pp), pp

all_ft_seqs = []
task_ids = []

for t in tasks:
    for e in eps:
        # pattern must contain this
        fname = path % (t, e)
        if args.allow_missing and not os.path.exists(fname):
            continue
        else:
            try:
                force_torques = np.load(fname)
            except ValueError as ve:
                print(fname, ve)
                raise ve
            all_ft_seqs.append(np.clip(force_torques, -args.clip, args.clip))
            task_ids.append(t)

# E lists (Ni, 3), for each
forces = [fseq[..., :3] for fseq in all_ft_seqs]
torques = [fseq[..., 3:] for fseq in all_ft_seqs]

fig = plt.figure(figsize=(10,6), tight_layout=True)
axes = fig.subplots(nrows=2, ncols=3)

cm = plt.get_cmap('Accent', len(tasks))
force_ax_labels = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
for r in range(2):
    for c in range(3):
        ax = axes[r][c]
        added_set = set()
        ax.set_title(force_ax_labels[r * 3 + c])
        for seq, t_id in zip(all_ft_seqs, task_ids):
            color = cm(tasks.index(t_id))
            # print(t_id, color)
            # print(seq.shape)
            if t_id in added_set:
                ax.plot(range(seq.shape[0]), seq[:, r * 3 + c], c=color, label="_")
            else:
                ax.plot(range(seq.shape[0]), seq[:, r * 3 + c], c=color, label=labels[tasks.index(t_id)])
            added_set.add(t_id)

        ax.legend()

plt.show()
