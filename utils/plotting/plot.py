import matplotlib.pyplot as plt

def plot_and_save_fig_4_2(X, Y1, Y2, x_label, y_label, title, xticks, filename):
    '''
    Y : 2D array sample energies over 500 iterations from 5 chains
    '''

    plt.figure(figsize=(20,7))
    #color_array = ['red', 'green', 'blue', 'yellow', 'cyan']

    
    plt.plot(X, Y1, 'o--', color="red")
    plt.plot(X, Y2, 'o--', color="green")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(xticks)
    
    plt.savefig(filename)
    plt.close()



train_ap = [61.3, 52.4, 53.7, 56.4, 62.7, 63.6, 66.4, 64.3, 65.3, 65.8, 65.5]
val_ap = [35.4, 28.6, 30.6, 29.4, 34.8, 34.6, 33.3, 33.7, 32.7, 33.7, 32.0]

plot_and_save_fig_4_2(range(50000,83000,3000), train_ap, val_ap, "Num iterations", "Avg MAP", "Train Val MAP(Train) vs Iterations", range(50000,83000,3000), "map_plot_train_val_iter2.png")