import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats

class PlotFigure:
    def __init__(self, figsize=(8, 6),fontname:str='Century Gothic',color_cycle=None):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.plots = []
        self.fontname = fontname
        self.ax.tick_params(axis='y')
        color_cycle 
        # plt.style.use('seaborn-white')
        if color_cycle is None:
            color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
            # color_cycle = plt.rcParams['axes.color_cycle'][2] #plt.get_cmap('Set1').colors # sns.color_palette("pastel")
        self.ax.set_prop_cycle('color', color_cycle)
    def add_histogram(self, data, bins=10, label='',normal_distribution=False, color=None):
        """
        Add a histogram to the figure.
        
        Parameters:
        - data: Data for the histogram.
        - bins: Number of bins for the histogram (default is 10).
        - label: Label for the histogram.
        - normal_distribution: If True, plot a normal distribution with the same mean and standard deviation as the data (default is False).
        - color: Color for the histogram (default is None).
        """
        self.plots.append(self.ax.hist(data, bins=bins, label=label, color=color))
        if normal_distribution:
            mean = data.mean()
            std = data.std()
            x = np.linspace(data.min(), data.max(), 100)
            y = stats.norm.pdf(x, mean, std)
            self.plots.append(self.ax.plot(x, y, label='Normal Distribution', color='red'))

    def add_plot(self, x, y, label='', linestyle='-',axis='y1', marker=None, color=None):
        """
        Add a plot to the figure.
        
        Parameters:
        - x, y: Data for the plot.
        - label: Label for the plot.
        - linestyle: Line style for the plot (default is '-').
        - marker: Marker for the plot points (default is None).
        - color: Color for the plot (default is None).
        """
        if axis == 'y2':
            if not hasattr(self, 'ax_sec'):
                self.add_secondary_yaxis()
                self.ax_sec.set_prop_cycle('color', self.ax._get_lines.prop_cycler.__next__()['color'])
            obj = self.ax_sec

        else: 
            obj = self.ax
        if marker == 'o':
            new_plot = obj.scatter(x, y, label=label, marker=marker, color=color)
        else:
            new_plot = obj.plot(x, y, label=label, linestyle=linestyle, marker=marker, color=color)
        self.plots.append(new_plot)

    def _set_scaling(self,):
        if not hasattr(self, 'ax_sec'):
            self.add_secondary_yaxis()
            # set same limits as primary y-axis
            limits1 = self.ax.get_ylim()
            self.ax_sec.set_ylim(limits1)

    def add_hline(self, y, label='', linestyle='-', color='black'):
        """
        Add a horizontal line to the figure.
        
        Parameters:
        - y: y-coordinate for the line.
        - label: Label for the line.
        - linestyle: Line style for the line (default is '-').
        - color: Color for the line (default is 'black').
        """
        self.plots.append(self.ax.axhline(y=y, label=label, linestyle=linestyle, color=color))
    def add_vline(self, x, label='', linestyle='-', color='black'):
        """
        Add a vertical line to the figure.
        
        Parameters:
        - x: x-coordinate for the line.
        - label: Label for the line.
        - linestyle: Line style for the line (default is '-').
        - color: Color for the line (default is 'black').
        """
        self.plots.append(self.ax.axvline(x=x, label=label, linestyle=linestyle, color=color))
    def set_title(self, title):
        """Set the title for the figure."""
        self.ax.set_title(title)

    def set_xlabel(self, xlabel):
        """Set the x-axis label for the figure."""
        self.ax.set_xlabel(xlabel,fontname=self.fontname)

    def set_ylabel(self, ylabel, axis='y1'):
        """Set the y-axis label for the figure."""
        if axis=='y2':
            if hasattr(self, 'ax_sec'):
                self.ax_sec.set_ylabel(ylabel,fontname=self.fontname)
            else: 
                raise ValueError('Secondary y-axis not added. Please add a secondary y-axis before setting the label.')
        else:
            self.ax.set_ylabel(ylabel,fontname=self.fontname)

    def add_legend(self):
        """Add a legend to the figure."""
        # create one common legend for all plots on both axes
        handles, labels = self.ax.get_legend_handles_labels()
        if hasattr(self, 'ax_sec'):
            handles_sec, labels_sec = self.ax_sec.get_legend_handles_labels()
            handles += handles_sec
            labels += labels_sec
        self.ax.legend(handles, labels,)
        
    def set_number_format(self, axis, format_string='{x:.2f}'):
        """
        Set number format for ticks on the specified axis.
        
        Parameters:
        - axis: The axis for which to set the number format ('x' for x-axis, 'y' for y-axis).
        - format_string: Format string specifying the number format. Default is '{x:.2f}' for floating point with 2 decimal places.
        """
        if axis == 'x':
            self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(format_string))
        elif axis == 'y':
            self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(format_string))
        else:
            print("Invalid axis. Please specify 'x' or 'y'.")
    def add_gridlines(self, axis='both', color='gray'):
        """
        Add gridlines to the plot.
        
        Parameters:
        - axis: Axis for which to add gridlines ('both', 'x', or 'y'). Default is 'both'.
        - color: Color of the gridlines. Default is 'gray'.
        """
        if axis == 'both':
            self.ax.grid(color=color,zorder=0)
        elif axis == 'x':
            self.ax.xaxis.grid(color=color,zorder=0)
        elif axis == 'y':
            self.ax.yaxis.grid(color=color,zorder=0)
        else:
            print("Invalid axis. Please specify 'both', 'x', or 'y'.")

    def add_secondary_yaxis(self, label='', color='tab:blue'):
        """
        Add a secondary y-axis to the plot that overlays the primary y-axis.
        
        Parameters:
        - label: Label for the secondary y-axis.
        - color: Color for the secondary y-axis.
        """
        self.ax_sec = self.ax.twinx()
        self.ax_sec.tick_params(axis='y')

    def show(self):
        """Display the figure."""
        plt.show()

    def save_figure(self, filename, dpi=300, format='png'):
        """
        Save the figure to a file.
        
        Parameters:
        - filename: Name of the file to save the figure to.
        - dpi: Resolution in dots per inch (default is 300).
        - format: File format (default is 'png').
        """
        self.fig.savefig(filename, dpi=dpi, format=format)