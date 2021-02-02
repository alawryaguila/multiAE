'''
Class for creating various plots

'''
from os.path import join
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
from sklearn.manifold import TSNE
import umap
from collections import OrderedDict
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.switch_backend('Agg')

class Plotting:
    def __init__(self, *path):
        if path:
            self.out_path = path

    def plot_losses(self, logger):
        plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.subplot(1,2,1)
        plt.title('Loss values')
        for k, v in logger.logs.items(): 
            plt.plot(v, label=str(k))
        plt.xlabel(r'\textbf{epochs}', fontsize=10)
        plt.ylabel(r'\textbf{loss}', fontsize=10)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title('Loss relative values')
        for k, v in logger.logs.items(): 
            max_loss = 1e-8 + np.max(np.abs(v)).cpu().detach().numpy()
            plt.plot(v/max_loss, label=str(k))
        plt.legend()
        plt.xlabel(r'\textbf{epochs}', fontsize=10)
        plt.ylabel(r'\textbf{loss}', fontsize=10)
        plt.savefig(join(self.out_path, "Losses.png"))
        plt.close()

    def plot_tsne(self, data, target, title, title_short):
        for i, data_ in enumerate(data):
            tsne_model = TSNE(n_components=2)
            projections = tsne_model.fit_transform(data_)
            color_code = ['b','r','c','m','y','g','k','orange', 'pink', 'gray']
            plt.figure()
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            for j in range(len(np.unique(target))):
                plt.plot(projections[np.where(target==j),0],projections[np.where(target==j),1], color=color_code[j], label = str(j), linestyle='None', marker = '.')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.title('t-SNE {0}'.format(title[i]))
            plt.savefig(join(self.out_path, 'tSNE_view_{0}.png'.format(title_short[i])))
            plt.close()

    def plot_UMAP(self, data, target, title, title_short):
        if len(data) != self.n_views:
            reducer = umap.UMAP()
            projections = reducer.fit_transform(data)
            color_code = ['b','r','c','m','y','g','k','orange', 'pink', 'gray']
            plt.figure()
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            for j in range(len(np.unique(target))):
                plt.plot(projections[np.where(target==j),0],projections[np.where(target==j),1], color=color_code[j], label = str(j), linestyle='None', marker = '.')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.title('UMAP {0}'.format(title))
            plt.savefig(join(self.out_path, 'UMAP_view_{0}.png'.format(title_short)))
            plt.close()  
        else:         
            for i, data_ in enumerate(data):   
                reducer = umap.UMAP()
                projections = reducer.fit_transform(data_)
                color_code = ['b','r','c','m','y','g','k','orange', 'pink', 'gray']
                plt.figure()
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                for j in range(len(np.unique(target))):
                    plt.plot(projections[np.where(target==j),0],projections[np.where(target==j),1], color=color_code[j], label = str(j), linestyle='None', marker = '.')
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                plt.title('UMAP {0}'.format(title[i]))
                plt.savefig(join(self.out_path, 'UMAP_view_{0}.png'.format(title_short[i])))
                plt.close()
