import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

__all__ = [
  'SimpleWatcher',
  'ExperimentWatcher'
]

try:
  from IPython import display
except ImportError:
  display = None

### http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=8
_default_color_seq = ('#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999')

class SimpleWatcher(object):
  def __init__(self, title='Watcher', labels=('loss', ), colors=_default_color_seq, figsize=(9, 6), save_dir='./'):
    self.save_dir = save_dir

    self.fig = plt.figure(figsize=figsize)
    self.ax = self.fig.add_subplot(111)

    self.ax.set_xlim([0.0, 1.0])
    self.ax.set_ylim([0.0, 1.0])

    self.fig.suptitle(title)
    self.title = title

    self.colors = colors
    self.labels = labels
    self.drawn = False

    self.data = None
    self.reset()

  def reset(self):
    self.data = [
      ([], [], [], [], []) for _ in range(len(self.labels))
    ]

  def _get_ylim(self):
    min_y = min([min(x_min) for x_min, _, _, _, _ in self.data])
    max_y = max([max(x_max) for _, _, _, _, x_max in self.data])

    delta = (max_y - min_y) / 2
    center = (max_y + min_y) / 2

    lower_bound = center - 1.05 * delta
    upper_bound = center + 1.05 * delta

    return lower_bound, upper_bound

  def update(self, *data):
    for d, new_data in zip(self.data, data):
      lower2, lower1, upper1, upper2 = np.percentile(new_data, q=(5, 20, 80, 95), axis=0)
      mean = np.mean(new_data)

      d[0].append(lower2)
      d[1].append(lower1)
      d[2].append(mean)
      d[3].append(upper1)
      d[4].append(upper2)

    return self

  def draw(self):
    self.ax.clear()
    display.clear_output(wait=True)

    x_lim = max([len(d[0]) for d in self.data])
    self.ax.set_xlim(0.0, x_lim)

    y_lower, y_upper = self._get_ylim()
    self.ax.set_ylim([y_lower, y_upper])

    for d, color, label in zip(self.data, self.colors, self.labels):
      lower2, lower1, mean, upper1, upper2 = d

      iters = np.arange(len(d[0])) + 0.5

      self.ax.fill_between(iters, lower1, upper1, alpha=0.2, color=color)
      self.ax.fill_between(iters, lower2, upper2, alpha=0.1, color=color)

      self.ax.plot(iters, mean, label=label, color=color)

    if not self.drawn:
      self.ax.legend()
      self.drawn = False

    display.display(self.fig)

    self.fig.savefig(osp.join(self.save_dir, '%s.png' % self.title), dpi=120)

class ExperimentWatcher(object):
  def __init__(self, title='learning curves', colors=_default_color_seq, figsize=(9, 6), save_path=None):
    self.save_path = save_path

    self.fig = plt.figure(figsize=figsize)
    self.ax = self.fig.add_subplot(111)

    self.ax.set_xlim([0.0, 1.0])
    self.ax.set_ylim([0.0, 1.0])

    self.fig.suptitle(title)
    self.title = title
    self.colors = colors

    self.drawn = False

    self.data = None

  def reset(self):
    self.data = None

  def _get_ylim(self):
    min_y = min([
      min(d) for name in self.data for d in self.data[name]
    ])

    max_y = max([
      max(d) for name in self.data for d in self.data[name]
    ])

    delta = (max_y - min_y) / 2
    center = (max_y + min_y) / 2

    lower_bound = center - 1.05 * delta
    upper_bound = center + 1.05 * delta

    return lower_bound, upper_bound

  def update(self, epoch_history):
    if self.data is None:
      self.data = dict([
        (name, [
          [] for _ in epoch_history[name]
        ]) for name in epoch_history
      ])

    for name in epoch_history:
      for history, d in zip(epoch_history[name], self.data[name]):
        d.append(np.mean(history))

    return self

  def draw(self):
    if self.data is None:
      return

    self.ax.clear()
    display.clear_output(wait=True)

    x_lim = max([
      max([ len(d) for d in self.data[name] ])
      for name in self.data
    ])
    self.ax.set_xlim(0.0, x_lim)

    y_lower, y_upper = self._get_ylim()
    self.ax.set_ylim([y_lower, y_upper])

    for name, color in zip(self.data, self.colors):
      for i, mean in enumerate(self.data[name]):
        iters = np.arange(len(mean)) + 0.5

        if i == 0:
          self.ax.plot(iters, mean, label=name, color=color)
        else:
          self.ax.plot(iters, mean, color=color)

    if not self.drawn:
      self.ax.legend()
      self.drawn = False

    display.display(self.fig)

    if self.save_path is not None:
      self.fig.savefig(self.save_path, dpi=120)