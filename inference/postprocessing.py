import numpy as np
import scipy.ndimage as nd


def keep_largest_component_per_class(mask, num_classes):

    final_mask = np.zeros_like(mask)

    for c in range(1, num_classes):

        binary = mask == c

        labeled, num = nd.label(binary)

        if num == 0:
            continue

        sizes = nd.sum(binary, labeled, range(1, num + 1))

        largest = np.argmax(sizes) + 1

        final_mask[labeled == largest] = c

    return final_mask
