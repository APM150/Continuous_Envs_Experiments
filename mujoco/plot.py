import numpy as np
import matplotlib.pyplot as plt
from components.filesys_manager import ExperimentPath


# overall return
def plot_stack(data, x: int, y: int, colors=None):
    """ x column is horizontal axis, y is vertical axis"""
    if colors is None:
        colors = [f"C{i}" for i in range(len(data))]
    for key, color in zip(data, colors):
        sp = data[key][0, :, x]
        # print(sp)
        values = data[key][:, :, y]
        mean, std = np.mean(values, axis=0), np.std(values, axis=0)
        if sp.shape[0] > 9:
            print('return', sp[9], mean[9])
        if sp.shape[0] > 49:
            print('return', sp[49], mean[49])
        # print(values, mean)
        # uncomment for aggr plot
        plt.plot(sp, mean, label=key, color=color)
        plt.fill_between(sp, mean - std, mean + std, color=color, alpha=.2)

        # uncomment for individual plot
        # for yi in values:
        #     plt.plot(sp, yi, label=key, color=color)

def plot(args):

    locals().update(args)
    for game in games:
        print(game)
        plt.title(f"{game} eval_return")
        plt.xlabel("transition sampled")
        plt.ylabel("episode return")
        datastack = exp[game].sync_stack(
            labels, [f"{label}/[0-9]*/eval_mean_stats*.csv" for label in labels]
            # ,morethan=38
        )
        plot_stack(datastack, 0, 3)
        plt.legend()
        plt.grid()
        exp_plot[game]['eval_return'].savefig()

    def arr_aggr(x, y):
        """ x: [1,1,2,3,3,4,4], y: [y1,y2,y3,y4,y5,y6,y7] -> """
        x_unique = np.unique(x.astype(int))
        res = np.zeros(x_unique.shape)
        for i, xi in enumerate(x_unique):
            res[i] = y[x == xi].mean()
        return x_unique, res

    # bias
    for game in games:

        plt.title(f"{game} eval bias")
        plt.xlabel("transition sampled")
        plt.ylabel("E_pi[V(s0)-R(xi)]")

        # plt.yscale('log')
        plt.ylim(-5,20)

        for label, color in zip(labels, colors):
            # print(game, label)
            paths = list(exp[game].iglob(f"{label}/*/eval_episode_stats*.csv"))

            data = []
            for path in paths:
                path_data = ExperimentPath(path).csv_read(nrows=None)
                x, y = path_data[:,0].astype(int), (path_data[:,1] - path_data[:,3])
                # plt.plot(x, y, label=label, color=color)
                data.append(arr_aggr(x, y))
            mints = min([len(x) for x, y in data])
            x = data[0][0][:mints]
            ys = np.stack([y[:mints] for _, y in data])
            y_mean, y_std = ys.mean(0), ys.std(0)

            plt.plot(x, y_mean, label=label, color=color)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)
        plt.plot([x[0],x[-1]], [0,0], color='black')
        plt.legend()
        plt.grid()
        exp_plot[game]['eval_bias'].savefig()

    # bias #2
    for game in games:

        plt.title(f"{game} eval bias")
        plt.xlabel("transition sampled")
        plt.ylabel("E_pi[V(s0)-R(xi)]")

        # plt.yscale('log')
        # plt.ylim(-5,20)

        for label, color in zip(labels, colors):
            # print(game, label)
            paths = list(exp[game].iglob(f"{label}/*/eval_episode_stats*.csv"))

            data = []
            for path in paths:
                path_data = ExperimentPath(path).csv_read(nrows=None)
                x, y = path_data[:,0].astype(int), (path_data[:,1] - path_data[:,3])
                # plt.plot(x, y, label=label, color=color)
                data.append(arr_aggr(x, y))
            mints = min([len(x) for x, y in data])
            x = data[0][0][:mints]
            ys = np.stack([y[:mints] for _, y in data])
            y_mean, y_std = ys.mean(0), ys.std(0)

            plt.plot(x, y_mean, label=label, color=color)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)
        plt.plot([x[0],x[-1]], [0,0], color='black')
        plt.legend()
        plt.grid()
        exp_plot[game]['eval_bias_nolim'].savefig()


    for game in games:

        # plt.yscale('log')
        # plt.ylim(-10,200)

        for label, color in zip(ensemble_labels, ensemble_colors):
            # print(label, color)

            paths = list(exp[game].iglob(f"{label}/*/debug/trace*.csv"))
            for runi, run in enumerate(paths):
                plt.title(f"{run}")
                plt.xlabel("transition sampled")
                plt.ylabel("V(s0)")
                path_data = ExperimentPath(run).csv_read(nrows=None)

                x, y = path_data[:,0], path_data[:,1:]
                for i in range(y.shape[1]):
                    xi, yi = arr_aggr(x, y[:,i])
                    plt.plot(xi, yi, color='blue', label='trace')
                plt.plot(*arr_aggr(x, y.mean(1)), color='red', label='mean')
                # plt.legend()
                plt.grid()
                exp_plot[game][f'{label}'][f'{runi}'].savefig()


if __name__ == '__main__':
    games = ['Humanoid-v2']
    labels = [
        'SAC', 'Mean_SAC', 'Mean_DDPG'
        ]
    ensemble_labels = [
        'SAC', 'Mean_SAC', 'Mean_DDPG'
    ]
    # paths
    exp = ExperimentPath("exp")["original"]
    exp_plot = ExperimentPath("exp_plot")["original"]

    colors = [f'C{i}' for i in range(len(labels))]
    ensemble_colors = [f'C{i}' for i in range(len(ensemble_labels))]

    plot(locals())
