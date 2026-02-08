import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter


if __name__ == "__main__":
    # Load data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    N, M = cmap.shape
    hf = HistogramFilter()
    belief = np.ones((N, M))

    num_steps = actions.shape[0]
    cols = 5
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(num_steps):
        belief = hf.histogram_filter(cmap, belief, actions[i], observations[i])

        ax = axes[i]
        # Show belief distribution as heatmap
        ax.imshow(belief, cmap='hot', interpolation='nearest', origin='upper')

        # Mark ground truth position
        gt_r, gt_c = belief_states[i]
        ax.plot(gt_c, gt_r, 'co', markersize=8, markeredgecolor='cyan', markeredgewidth=2, label='GT')

        # Mark estimated position (argmax of belief)
        est_r, est_c = np.unravel_index(np.argmax(belief), belief.shape)
        ax.plot(est_c, est_r, 'g^', markersize=8, markeredgecolor='lime', markeredgewidth=2, label='Est')

        ax.set_title(f'Step {i}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for j in range(num_steps, len(axes)):
        axes[j].axis('off')

    # Add shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, label='Ground Truth'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', markersize=10, label='Estimate'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11)

    fig.suptitle('Histogram Filter: Belief Heatmap per Step', fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig('data/belief_heatmap.png', dpi=150)
    plt.show()
    print("Saved to data/belief_heatmap.png")
