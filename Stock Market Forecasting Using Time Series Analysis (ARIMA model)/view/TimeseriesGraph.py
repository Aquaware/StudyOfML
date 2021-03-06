# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from datetime import datetime
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()


YMD = 'YMD'
MD = 'MD'
YMDHM = 'YMDHM'
MDHM = 'MDHM'
HM = 'HM'

time_labels = {YMDHM: '%Y-%m-%d %H:%M', YMD: '%Y-%m-%d', MDHM: '%m-%d %H:%M', MD: '%m-%d', HM: '%H:%M'}



def toNaive(time_list):
    out = []
    for time in time_list:
        naive = datetime(time.year, time.month, time.day, time.hour, time.minute, time.second)
        out.append(naive)
    return out


class TimeseriesGraph(object):
    def __init__(self, ax, aware_pytime, time_label_type):
        self.ax = ax
        self.twin_ax = ax.twinx()
        self.pytime = aware_pytime
        time_list = toNaive(aware_pytime)
        self.time = mdates.date2num(time_list)
        self.time_label_type = time_label_type
        self.length = len(aware_pytime)

    def grid(self):
        self.ax.grid()
        return

    def setTitle(self, title, xlabel, ylabel):
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        pass

    def scatter(self, value, color_index, line_width):
        self.ax.scatter(self.time, value, c=self.color(color_index), lw=line_width)
        pass

    def plot(self, value, style_index, line_width):
        sty = self.style(style_index)
        self.ax.plot(self.time, value, color=sty[0], linestyle=sty[1], lw=line_width)
        self.ax.grid()
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(time_labels[self.time_label_type]))  # '%m-%d %H:%M'))
        pass

    def plotFlag(self, flag, values, props):
        if len(flag) != len(self.time) or len(values) != len(self.time):
            return
        for i in range(len(flag)):
            for prop in props:
                if prop['value'] == flag[i]:
                    marker = prop['marker']
                    color = prop['color']
                    alpha = prop['alpha']
                    size = prop['size']
                    self.point([self.time[i], values[i]], marker, color, alpha, size)
        return

    def box(self, xrange, yrange, color_index, alpha):
        if yrange is None:
            bottom, top = self.ax.get_ylimi()
            d = yrange[0] - bottom
            y0 = d / (top - bottom)
            d = yrange[1] - bottom
            y1 = d / (top - bottom)
        else:
            y0 = yrange[0]
            y1 = yrange[1]
        self.ax.axvspan(xrange[0], xrange[1], y0, y1, color=self.color(color_index), alpha=alpha)
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(time_labels[self.time_label_type]))  # '%m-%d %H:%M'))
        pass

    def point(self, point, marker, color, alpha, size):
        self.ax.scatter(point[0], point[1], s=size, alpha=alpha, marker=marker, c=color)
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(time_labels[self.time_label_type]))  # '%m-%d %H:%M'))
        pass

    def xLimit(self, xrange):
        self.ax.set_xlim(xrange[0], xrange[1])

    def yLimit(self, yrange):
        self.ax.set_ylim(yrange[0], yrange[1])
        pass

    def xRange(self):
        return self.ax.get_xlim()

    def yRange(self):
        return self.ax.get_ylim()

    def drawLegend(self, lines, markers):
        elements = []
        if markers is not None:
            for marker in markers:
                e = Line2D([0], [0], marker=marker['marker'], color=marker['color'], linewidth=0, label=marker['label'],
                           markersize=marker['size'] / 10)
                elements.append(e)
        if lines is not None:
            for line in lines:
                if type(line['color']) is int:
                    c = self.color(line['color'])
                else:
                    c = line['color']
                e = Line2D([0], [0], marker='o', color=c, linewidth=5, label=line['label'],
                           markerfacecolor=line['color'], markersize=0)
                elements.append(e)
        self.ax.legend(handles=elements,loc='upper left', borderaxespad=0)
        pass

    def markingWithFlag(self, y, mark_flag, markers):
        for marker in markers:
            for i in range(len(y)):
                if mark_flag[i] == marker['status']:
                    self.point([self.time[i], y[i]], marker['marker'], marker['color'], marker['alpha'], marker['size'])
        pass

    def markingWithTime(self, values, mark_time, marker):
        if mark_time is None:
            return
        if len(mark_time) == 0:
            return
        times = mdates.date2num(mark_time)
        for tpy, v in zip(self.pytime, values):
            for tmark, time in zip(mark_time, times):
                if tmark == tpy:
                    self.point([time, v], marker['marker'], marker['color'], marker['alpha'], marker['size'])
        pass

    def hline(self, values, colors, width):
        for i in range(len(values)):
            value = values[i]
            if len(values) == len(colors):
                c = colors[i]
            else:
                c = colors[0]
            y = np.full(self.length, value)
            self.ax.plot(self.time, y, color=c, linewidth=width)
        pass

    def vline(self, values, colors, width):
        for i in range(len(values)):
            value = values[i]
            if len(values) == len(colors):
                c = colors[i]
            else:
                c = colors[0]
            x = np.full(2, value)
            y = self.ax.get_ylim()
            self.ax.plot(x, y, color=c, linewidth=width)
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(time_labels[self.time_label_type]))  # '%m-%d %H:%M'))
        pass

    def text(self, x, y, text, color, size):
        self.ax.text(x, y, text, color=color, size=size)
        self.twin_ax.xaxis_date()
        self.twin_ax.xaxis.set_major_formatter(
            mdates.DateFormatter(time_labels[self.time_label_type]))  # '%m-%d %H:%M'))
        return

    def bars(self, values, colors, limit, width):
        if len(values) != self.length or len(colors) != self.length:
            print('bars ... error')
            return
        for t, v, c in zip(self.time, values, colors):
            self.bar(t, v, c, width)
        self.twin_ax.set_ylim(0, limit)
        return

    def bar(self, x, value, color, width):
        self.twin_ax.xaxis.set_major_formatter(mdates.DateFormatter(time_labels[self.time_label_type]))
        self.twin_ax.grid()
        self.twin_ax.xaxis_date()
        self.twin_ax.xaxis.set_major_formatter(
            mdates.DateFormatter(time_labels[self.time_label_type]))  # '%m-%d %H:%M'))
        return

    @classmethod
    def makeFig(cls, rows, cols, size):
        fig, ax = plt.subplots(rows, cols, figsize=(size[0], size[1]))
        return (fig, ax)

    @classmethod
    def gridFig(cls, width, heights):
        rows = len(heights)
        height = 0
        for h in heights:
            height += h

        ratios = []
        for i in range(rows):
            if i == 0:
                ratio = 1.0
            else:
                ratio = heights[i] / heights[0]
            ratios.append(ratio)

        axes = []
        fig = plt.figure(figsize=(width, height))
        grids = GridSpec(nrows=rows, ncols=1, height_ratios=ratios)
        for i in range(rows):
            grid = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grids[i, 0])
            ax = fig.add_subplot(grid[:, :])
            axes.append(ax)
        return (fig, axes)

    @classmethod
    def color(cls, index):
        colors = [mcolors.CSS4_COLORS['red'], mcolors.CSS4_COLORS['blue'], mcolors.CSS4_COLORS['green'],
                  mcolors.CSS4_COLORS['magenta'], mcolors.CSS4_COLORS['pink'], mcolors.CSS4_COLORS['gold'],
                  mcolors.CSS4_COLORS['orangered'],
                  mcolors.CSS4_COLORS['yellowgreen'], mcolors.CSS4_COLORS['cyan'], mcolors.CSS4_COLORS['darkgrey'],
                  mcolors.CSS4_COLORS['blue']]
        return colors[int(index % len(colors))]

    @classmethod
    def lineStyle(cls, index):
        array = ['solid', 'dashed', 'dashdot']
        return array[int(index % len(array))]

    @classmethod
    def style(cls, index):
        c = cls.color(index)
        style = cls.lineStyle(int(index / 10))
        return [c, style]