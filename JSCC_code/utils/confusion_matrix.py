import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
from evaluation.evaluation import get_y_preds


def confusion_matrix(predictions, gt, class_num, output_file=None):
    # Plot confusion_matrix and store result to output_file
    class_names = range(class_num)
    predictions = get_y_preds(gt, predictions, class_num)
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)

    plt.imshow(confusion_matrix, cmap='Oranges')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='black', fontsize=10)
        else:
             pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()