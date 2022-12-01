from __future__ import print_function, division

from collections import OrderedDict
from collections import defaultdict
from os.path import commonprefix
import itertools as it
import json
import textwrap

import numpy as np
from plotly.graph_objs import Figure
from plotly.graph_objs import Layout
from plotly.graph_objs.layout import Margin
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from plotly.offline import plot
from plotly.offline.offline import get_plotlyjs

import os
import os.path as pt

COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]
HTML_BEGIN = ''.join([
    '<html><head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/'
    'bootstrap/3.3.1/css/bootstrap.min.css"><style>body{ margin:100px; backgro'
    'und:whitesmoke; }</style><script src="plotly.js"></script></head><body>'
])
HTML_H1 = '<h1>%s</h1>'
HTML_END = '</body></html>'


def get_logfiles(plotdirs):
    names = []
    paths = []
    for d in plotdirs:
        if pt.isfile(d):
            if not d.endswith('.log'):
                continue
            paths.append(d)
            names.append(pt.splitext(pt.basename(d))[0])
        else:
            for sub in os.listdir(d):
                sub = pt.join(d, sub)
                files = sorted([
                    f for f in os.listdir(sub) if f.endswith('.log')
                ])
                for f in files:
                    f = pt.join(sub, f)
                    paths.append(f)
                    names.append(pt.basename(f.replace('.log', '')))
    return zip(*sorted(zip(names, paths)))


def _get_data_legacy(s):
    parts = s.split(' ')
    try:
        v = float(parts[-1])
    except ValueError:
        return {}
    if s.startswith('train'):
        return {'phase': 'train', 'metrics': {'loss': v}}
    elif s.startswith('val'):
        return {'phase': 'val', 'metrics': {'top-1 acc': v}}
    else:
        return {}


def _append_values(d, s, e):
    for k, v in s.items():
        if e is None or len(d[k]) <= e:
            d[k].append([v])
        else:
            d[k][e].append(v)


def get_data(paths):
    x = []
    val_lists = defaultdict(lambda: [])
    train_lists = defaultdict(lambda: [])
    for path in paths:
        with open(path, 'r') as f:
            fdata = [line.rstrip() for line in f]
        for line in fdata:
            _, found, msg = line.rpartition(' - ')
            try:
                data = json.loads(msg)
            except ValueError:
                data = _get_data_legacy(msg)
            try:
                epoch = data.get('epoch', len(x) + 1)
                if data.get('phase') == 'train':
                    _append_values(train_lists, data['metrics'], x.index(epoch) if epoch in x else None)
                    if epoch not in x:
                        x.append(epoch)
                elif data.get('phase') == 'val':
                    _append_values(val_lists, data['metrics'], x.index(epoch) if epoch in x else None)
                    if epoch not in x:
                        x.append(epoch)
            except AttributeError:
                pass
    try:
        n1 = max(map(len, train_lists.values()))
    except ValueError:
        n1 = 0
    try:
        n2 = max(map(len, val_lists.values()))
    except ValueError:
        n2 = 0
    n = max(n1, n2)
    val_lists = {k: np.array([{'mean': np.mean(sv), 'std': np.std(sv)} for sv in v]) for k, v in val_lists.items()}
    train_lists = {k: np.array([{'mean': np.mean(sv), 'std': np.std(sv)} for sv in v]) for k, v in train_lists.items()}
    x = np.array(x, dtype=np.float64)
    return n, x, val_lists, train_lists


def path2name(path):
    return pt.basename(pt.dirname(path)) \
        .replace('.log', '')\
        .replace('_', ' ')


def remove_prefix(names):
    names = list(names)
    prefix = commonprefix(names)
    short_names = ['']
    while any(len(s) < 3 for s in short_names):
        prefix = ' '.join(prefix.split(' ')[:-1])
        if prefix:
            short_names = [n.rpartition(prefix)[2].strip() for n in names]
        else:
            short_names = names
    return prefix, short_names


def hex2rgb(hex):
    return tuple(int(c, 16) for c in textwrap.wrap(hex.lstrip('#'), 2))


def rgbcolor(c):
    try:
        return hex2rgb(c)
    except (AttributeError, ValueError):
        return c


def color_derivatives(c, n, delta=0.2):
    if n == 1:
        return c,
    c = np.array(rgbcolor(c))
    d = n/2 * delta
    scales = np.linspace(-d, d, n) + 1
    ds = (tuple(np.clip(c*s, 0, 255).tolist()) for s in scales)
    return tuple('rgb(%d, %d, %d)' % dc for dc in ds)


def _make_plot(x, y, name, phase, metric, color, style, std):
    pall = Scatter(
        x=x, y=np.array([d['mean'] for d in y]), line=Line(width=1.5, color=color, dash=style),
        name='%s %s %s' % (name, phase, metric), mode='lines'
    )
    pall_std_m = Scatter(
        x=x, y=np.array([d['mean'] - d['std'] for d in y]), line=Line(width=0.0, color=color, dash=style),
        name='%s %s %s' % (name, phase, metric), mode='lines', hoverinfo='skip',
    ) if std else None
    pall_std_p = Scatter(
        x=x, y=np.array([d['mean'] + d['std'] for d in y]), line=Line(width=0.0, color=color, dash=style),
        name='%s %s %s' % (name, phase, metric), mode='lines', fill='tonexty', hoverinfo='skip'
    ) if std else None
    psingle = Scatter(
        x=x, y=np.array([d['mean'] for d in y]), line=Line(width=1.5, color=color, dash=style),
        name='%s %s' % (phase, metric), mode='lines'
    )
    psingle_std_m = Scatter(
        x=x, y=np.array([d['mean'] - d['std'] for d in y]), line=Line(width=0.0, color=color, dash=style),
        name='%s %s %s' % (name, phase, metric), mode='lines', hoverinfo='skip'
    ) if std else None
    psingle_std_p = Scatter(
        x=x, y=np.array([d['mean'] + d['std'] for d in y]), line=Line(width=0.0, color=color, dash=style),
        name='%s %s %s' % (name, phase, metric), mode='lines', fill='tonexty', hoverinfo='skip'
    ) if std else None
    return pall, pall_std_m, pall_std_p, psingle, psingle_std_m, psingle_std_p


def _make_plots(name, x, train_lists, val_lists, colors, ymult, std, metrics):
    train_plots = []
    val_plots = []
    plots = []
    colors_derivs = color_derivatives(next(colors), len(train_lists))
    for c, (metric, values) in zip(colors_derivs, train_lists.items()):
        if len(metrics) == 0 or metric in metrics:
            ym = next(ymult)
            pall, pall_m, pall_p, psingle, psingle_m, psingle_p = _make_plot(
                x, np.array([{k: v*ym for k, v in d.items()} for d in values]), name, 'train', metric, c, '1px', std
            )
            train_plots.append(pall)
            if std:
                train_plots.append(pall_m)
                train_plots.append(pall_p)
            plots.append(psingle)
            if std:
                plots.append(psingle_m)
                plots.append(psingle_p)
    # colors_derivs = color_derivatives(next(colors), len(val_lists))
    for c, (metric, values) in zip(colors_derivs, val_lists.items()):
        if len(metrics) == 0 or metric in metrics:
            ym = next(ymult)
            pall, pall_m, pall_p, psingle, psingle_m, psingle_p = _make_plot(
                x, np.array([{k: v*ym for k, v in d.items()} for d in values]), name, 'val', metric, c, 'solid', std
            )
            val_plots.append(pall)
            if std:
                val_plots.append(pall_m)
                val_plots.append(pall_p)
            plots.append(psingle)
            if std:
                plots.append(psingle_m)
                plots.append(psingle_p)
    return train_plots, val_plots, plots


def main(dirs, outfile, xmult, ymult, std, metrics):
    xmult = it.cycle(xmult)
    ymult = it.cycle(ymult)
    outdir = pt.dirname(outfile)
    with open(pt.join(outdir, 'plotly.js'), 'w') as f:
        f.write(get_plotlyjs())
    names, paths = get_logfiles(dirs)
    prefix, names = remove_prefix(names)
    layout = Layout(
        width=1000,
        height=500,
        margin=Margin(b=100, t=100),
    )
    config = dict(
        output_type='div',
        include_plotlyjs=False,
        show_link=False,
    )

    all_plots = OrderedDict()
    train_plots = []
    val_plots = []
    colors = it.cycle(COLORS)
    curname = names[0]
    ps = []
    for count, np in enumerate(zip(names, paths)):
        name, p = np
        if curname == name:
            ps.append(p)
            if count + 1 < len(names):
                continue
        n, x, val_lists, train_lists = get_data(ps)
        if not n:
            continue
        print('%30s:%3d epochs' % (curname, n))
        x *= next(xmult)
        new_train_plots, new_val_plots, plots = _make_plots(
            curname, x, train_lists, val_lists, colors, ymult, std, metrics
        )
        train_plots.extend(new_train_plots)
        val_plots.extend(new_val_plots)
        fig = Figure(data=plots, layout=Layout(arg=layout, title=curname))
        all_plots[curname] = plot(fig, **config)
        curname = name
        ps = [p]

    # train plot
    fig = Figure(data=train_plots, layout=Layout(arg=layout, title='Train'))
    all_plots['Train'] = plot(fig, **config)

    # val plot
    fig = Figure(data=val_plots, layout=Layout(arg=layout, title='Val'))
    all_plots['Val'] = plot(fig, **config)

    # reorder
    all_plots = OrderedDict([
        ('Train', all_plots.pop('Train')),
        ('Val', all_plots.pop('Val')),
    ] + list(all_plots.items()))

    body = ''
    for content in all_plots.values():
        body += content

    html_string = ''.join(it.chain(
        [HTML_BEGIN],
        [HTML_H1 % prefix],
        all_plots.values(),
        [HTML_END],
    ))
    with open(outfile, 'w') as f:
        f.write(html_string)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'dirs',
        nargs='+',
        help='search logfiles in these directories',
    )
    parser.add_argument(
        'outfile',
        help='write to this plot file',
    )
    parser.add_argument(
        '--xmult',
        default=[1],
        type=float,
        nargs='+',
        help='multiplier for x-coordinates',
    )
    parser.add_argument(
        '--ymult',
        default=[1],
        type=float,
        nargs='+',
        help='multiplier for y-coordinates',
    )
    parser.add_argument(
        '--std',
        action="store_true",
        dest="std",
        help='activate to plot std deviation for multiple runs'
    )
    parser.add_argument(
        '--metrics', '-m',
        nargs='+',
        default=[],
        help='metrics to plot (e.g. pick of {loss, top-1 acc, ..}), defaults to all available (empty list)'
    )
    parser.set_defaults(
        std=False
    )
    args, unknown = parser.parse_known_args()
    if not pt.exists(args.outfile):
        os.makedirs(os.path.dirname(args.outfile))
    try:
        main(
            args.dirs,
            args.outfile,
            args.xmult,
            args.ymult,
            args.std,
            args.metrics,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
