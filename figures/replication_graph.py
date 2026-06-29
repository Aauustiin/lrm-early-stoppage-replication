import matplotlib.pyplot as plt
import numpy as np

categories = ['Explicit Reasoning', 'Coconut', 'CODI']
# original = [98, 54, 44]
# # replication = [86.60529260984151, 49.33030073287844, 41.546626231993933]
# replication = [86.6, 49.3, 41.5]  # rounded for display

original = [99, 69, 54]
# replication = [99.04148164193654, 63.07808946171342, 55.93884255749305]
replication = [99.0, 63.1, 55.9]  # rounded for display

x = np.arange(len(categories))
width = 0.38

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, original, width, label='original',
               facecolor='white', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, replication, width, label='replication',
               facecolor='white', edgecolor='black', linewidth=1.5,
               hatch='///')

# value labels on top of each bar
for bar, val in zip(list(bars1) + list(bars2), original + replication):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f'{val}%', ha='center', va='bottom', fontsize=10)

ax.set_ylim(0, 110)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_xticks(x)
ax.set_xticklabels(categories)
# ax.set_title('First Match')
ax.set_title('Stable Match')
ax.legend(loc='upper right', frameon=True, edgecolor='black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# plt.savefig('first_match.png', dpi=150)
plt.savefig('stable_match.png', dpi=150)
plt.show()