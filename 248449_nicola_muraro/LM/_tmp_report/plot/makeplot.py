import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    indice_rete = 2
    data = pd.read_csv('./data/network_' + str(indice_rete) + '.csv')
    fig1 = plt.figure(figsize=(10, 6))
    ax = fig1.add_subplot(111)
    sns.lineplot(x='epoch', y='train_ppl', data=data, ax=ax, label='train_loss')
    sns.lineplot(x='epoch', y='dev_ppl', data=data, ax=ax, label='dev_ppl')
    plt.savefig('./results/plot_' + str(indice_rete) + '.png')
