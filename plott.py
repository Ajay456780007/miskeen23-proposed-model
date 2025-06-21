# ---------------------------Importing Required Libraries----------------------#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns
import os

rcParams['font.weight'] = "bold"
rcParams['font.size'] = 20


class PlotAll:
    def __init__(self, show=False, save=False, **kwargs):
        self.show = show
        self.save = save
        self.rc_params = {
            'font.weight': kwargs.get("fontweight", "bold"),
            'font.size': kwargs.get("fontsize", 16)
        }
        rcParams.update(self.rc_params)
        self.legends = kwargs.get('legends', [])
        self.colors = kwargs.get('color', [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000',
            '#000080', '#808000', '#800080', '#008080', '#808080', '#C0C0C0', '#c658cc'
        ])
        self.fig_size = kwargs.get('fig_size', (10, 8))
        self.markers = kwargs.get('markers',
                                  ['o', 'p', '*', 'h', 'v', 'x', '.', '>', '<', 'p', 'o', '*', 'h', 'v', 'x'])

    def _save_dataframe(self, df, path, filename):
        df.to_csv(os.path.join(path, f'{filename}.csv'))

    def _save_plot(self, path, filename):
        plt.savefig(os.path.join(path, f'{filename}.png'), dpi=800)

    def _configure_plot(self, ax, xlab, ylab, loca, n_col, xtick):
        leg_prop = {'size': 20, 'weight': 'bold'}
        ax.legend(self.legends, fancybox=True, framealpha=0.5, shadow=False, borderpad=1, loc=loca, ncol=n_col,
                  prop=leg_prop)
        ax.set_xlabel(xlab, fontsize=20, fontweight='bold')
        ax.set_ylabel(ylab, fontsize=20, fontweight='bold')
        plt.xticks(ticks = [i for i in range(len(xtick))], labels=xtick)



    def line_plot(self, dat, xlab, ylab, loca='lower right', **kwargs):
        line_width = kwargs.get('linewidth', 1.5)
        mark_size = kwargs.get('markersize', 8)
        n_col = kwargs.get('n_col', 1)
        xticks = kwargs.get('xticks')
        df = pd.DataFrame(dat.T)
        try:
            df.columns = self.legends
            df.index = [f"TP_{i}" for i in xticks]
        except:
            df = df.T
            df.columns = self.legends
            df.index = [f"TP_{i}" for i in xticks]

        sns.set_theme(style="darkgrid")
        if self.save:
            self._save_dataframe(df, kwargs.get('path'), kwargs.get('filename'))

        ax = df.plot(kind='line', rot=0, linewidth=line_width, markersize=mark_size, figsize=self.fig_size)
        for i, line in enumerate(ax.get_lines()):
            line.set_marker(self.markers[i % len(self.markers)])

        self._configure_plot(ax, xlab, ylab, loca, n_col, xticks)

        if self.save:
            self._save_plot(kwargs.get('path'), kwargs.get('filename'))
        if self.show:
            plt.show()

    def bar_plot(self, dat, xlab, ylab, loca='lower right', **kwargs):
        edge_color = kwargs.get('edgecolor', 'black')
        bar_width = kwargs.get('bar_width', 0.8)
        n_col = kwargs.get('n_col', 2)
        xticks = kwargs.get('xticks')
        df = pd.DataFrame(dat.T)
        try:
            df.columns = self.legends
            df.index = [f"TP_{i}" for i in xticks]
        except:
            df = df.T
            df.columns = self.legends
            df.index = [f"TP_{i}" for i in xticks]

        sns.set_theme(style="darkgrid")

        if self.save:
            self._save_dataframe(df, kwargs.get('path'), kwargs.get('filename'))

        ax = df.plot(kind='bar', rot=0, figsize=self.fig_size, edgecolor=edge_color, width=bar_width)

        self._configure_plot(ax, xlab, ylab, loca, n_col, xticks)
        if self.save:
            self._save_plot(kwargs.get('path'), kwargs.get('filename'))
        if self.show:
            plt.show()


    def box_plot(self, dat, xlab, ylab, loca='lower right', **kwargs):
        n_col = kwargs.get('n_col', 1)
        df = pd.DataFrame(dat.T)
        xticks = kwargs.get('xticks')
        if self.save:
            self._save_dataframe(df, kwargs.get('path'), kwargs.get('filename'))

        ax = plt.figure(figsize=(10, 8))
        ax = sns.stripplot(data=df)
        ax = sns.boxplot(data=df)
        self._configure_plot(ax, xlab, ylab, loca, n_col, xticks)
        if self.save:
            self._save_plot(kwargs.get('path'), kwargs.get('filename'))
        if self.show:
            plt.show()


# # Example usage
# per = np.random.uniform(50, 100, size=(6, 5))
# leg = ["1", "2", "3", "4", "5", "6"]
#
# p = PlotAll(save=False, legends=leg)
#
# p.line_plot(per, "Training Percentage", "Accuracy(%)")
# p.bar_plot(per, "Training Percentage", "Accuracy(%)")
# p.box_plot(per, "Training Percentage", "Accuracy(%)")
