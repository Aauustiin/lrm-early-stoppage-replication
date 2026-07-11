import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

models = ['coconut', 'codi', 'sft']
labels = ['COCONUT', 'CODI', 'ERM']

proportions = {m: {} for m in models}
for name in models:
    with open(f'results/{name}.json') as f:
        data = json.load(f)
    proportions[name]['Original'] = data['slicing_original_proportion']
    proportions[name]['Augmented'] = data['slicing_augmented_proportion']
    proportions[name]['Other'] = data['slicing_other_proportion']
    proportions[name]['Tie'] = data['slicing_tie_proportion']

categories = ['Original', 'Augmented', 'Other', 'Tie']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

x = np.arange(len(models))
bar_width = 0.5

fig, ax = plt.subplots(figsize=(7, 5))

bottoms = np.zeros(len(models))
bars = []
for cat, color in zip(categories, colors):
    vals = np.array([proportions[m][cat] for m in models])
    b = ax.bar(x, vals, bar_width, bottom=bottoms, color=color, label=cat)
    bars.append(b)
    # Annotate segments that are large enough to label
    for i, (v, bot) in enumerate(zip(vals, bottoms)):
        if v >= 0.04:
            ax.text(
                x[i], bot + v / 2,
                f'{v:.1%}',
                ha='center', va='center',
                fontsize=14, color='white'
            )
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.set_ylabel('Proportion', fontsize=14)
ax.set_title('Token Patching Answer Status by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(categories),
          framealpha=0.9, fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/slicing_status.png', dpi=150)
print('Saved to results/slicing_status.png')
