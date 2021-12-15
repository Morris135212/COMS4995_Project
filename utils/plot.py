import matplotlib.pyplot as plt


def pr_curve(precision, recall):
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(recall, precision)
    ax.plot([0, 1], [1, 0], 'r--')
    ax.set_xlabel("recall", fontsize=15)
    ax.set_ylabel("precision", fontsize=15)
    plt.show()


def roc_curve(fpr, tpr):
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel("False positive", fontsize=15)
    ax.set_ylabel("True positive", fontsize=15)
    plt.show()


def pr_curve_mul(precisions, recalls, labels):
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    for prec, recall, label in zip(precisions, recalls, labels):
        ax.plot(recall, precisions, labels=label)
    ax.plot([0, 1], [1, 0], 'r--')
    ax.set_xlabel("recall", fontsize=15)
    ax.set_ylabel("precision", fontsize=15)
    plt.show()


def roc_curve_mul(fprs, tprs, labels):
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    for fpr, tpr, label in zip(fprs, tprs, labels):
        ax.plot(fpr, tpr, labels=label)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel("False positive", fontsize=15)
    ax.set_ylabel("True positive", fontsize=15)
    plt.show()
