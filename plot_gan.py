try:
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
except ImportError:
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

import sys
import numpy as np 
from sklearn.manifold import TSNE
import re
import pylab

def plot_metrics(perf_dict, savename=None, metric_name="Loss",
                title=None, itv=None, marker_size=5):
    
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel(metric_name)

    plt.title("%s Plot" % metric_name)
    # X = np.arange(len(list(perf_dict.values())[0]))
    for k, v in perf_dict.items():
        plt.plot(v[:, 0], v[:, 1], label=k)
    
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except:
        pass
    if savename is not None:
        plt.savefig(savename)
        print("Losses plot saved to %s" % savename)

def plot_latent_encodings(latent_encodings, labels, savename=None):
    plt.figure()
    plt.title("Latent Encodings")
    tsne = TSNE()
    latent_tsne = tsne.fit_transform(latent_encodings)
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap=pylab.cm.plasma)
    plt.xlabel("L1")
    plt.ylabel("L2")
    #plt.legend()
    try:
        plt.show()
    except:
        pass
    if savename is not None:
        plt.savefig(savename)
        print("Latent encoding plots saved to %s" % savename)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = sys.argv[1]
        X = np.load(f)
        plot_losses(X)
    if len(sys.argv) > 2:
        f = sys.argv[2]
        X = np.load(f)
        plot_latent_encodings(X)

