import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    print("belief_states: \n", belief_states)
    print("belief_states shape:", belief_states.shape)

    print("cmap: \n", cmap)
    print("cmap shape:", cmap.shape)
    
    print("actions: \n", actions)
    print("observations:", observations)

    filter = HistogramFilter()
    belief = np.ones((cmap.shape[0], cmap.shape[1]))
    for i in range(actions.shape[0]):
        belief = filter.histogram_filter(cmap, belief, actions[i], observations[i])
        belief_rot = np.rot90(belief, -1)
        max_index = np.unravel_index(np.argmax(belief_rot), belief_rot.shape)
        print(max_index)