import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    indice_rete = 1
    data = pd.read_csv('./data/network_' + str(indice_rete) + '.csv')
    fig1 = plt.figure(figsize=(7, 4))
    ax = fig1.add_subplot(111)
    #fig1.suptitle('Network ' + str(indice_rete), fontsize=16)
    sns.lineplot(x='epoch', y='train_ppl', data=data, ax=ax, label='Train')
    sns.lineplot(x='epoch', y='dev_ppl', data=data, ax=ax, label='Dev')
    ax.legend(fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PPL', fontsize=12)
    fig1.tight_layout()
    plt.savefig('./results/plot_' + str(indice_rete) + '.png')
