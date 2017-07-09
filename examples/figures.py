# function for plotting and saving figures
from __future__ import division, print_function
import numpy as np
import os

# babusca import via context
from context import smatrix
from context import g2 as g2calc

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

import matplotlib.ticker as ticker

from itertools import cycle

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
# mpl.rcParams['font.family'] = 'Helvetica'


def g1_coherent(ses, chli, chlo, d1s, directory, offset, ylim=None):
    """
    Calculate, plot and save g2 calculation
    """
    # defautls
    if ylim is None:
        ylim = ([1e-6, 1e6])

    # init
    g1s = np.zeros((len(ses), len(d1s)))

    print('calc: init')

    for i, se in enumerate(ses):
        _, S1 = smatrix.one_particle(se, chli, chlo, d1s + offset)
        g1s[i, :] = np.abs(S1) ** 2

    print('calc: done')
    print('layout: init')

    f, (ax1) = plt.subplots(1, figsize=(6, 3))
    # AX1

    # limits
    ax1.set_xlim([np.min(d1s), np.max(d1s)])
    ax1.set_ylim(ylim)

    # horizontal unit line
    ax1.semilogy(d1s, np.ones(d1s.size), linewidth=1, color='#BBBBBB')
    # res
    for i, se in enumerate(ses):
        ax1.semilogy(d1s, g1s[i, :], linewidth=1.2, label=se.label)

    ax1.legend(loc="center left", fancybox=False, fontsize=15, edgecolor=None, frameon=False, bbox_to_anchor=(1, 0.5))
    ax1.set_xlim([np.min(d1s), np.max(d1s)])
    ax1.set_ylim(ylim)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # locators
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    # labels
    ax1.set_ylabel(r'$g^{{(1)}}_{{{ins},{outs}}}$'.format(
        ins=chli,
        outs=chlo))
    ax1.set_xlabel(r'$\delta^{(1)}$ (units of $\Gamma$)')

    # update font size
    update_font(ax1, 16)

    # ax2
    # plt.tight_layout()
    print('layout: done')

    print('save: init')
    plt.savefig(directory + 'g1_{ins}{outs}_coherent_tau0.pdf'.format(ins=chli, outs=chlo),
                bbox_inches='tight')
    plt.close(f)

    print('save: done')


def g2_coherent(ses, chlsi, chlso, d1s, d2s, directory, offset, ylims=None, yticks=2, yticks2=None):
    """
    Calculate, plot and save g2 calculation
    """
    # defautls
    if ylims is None:
        ylim1 = ([1e-6, 1e6])
        ylim2 = ([1e-6, 1e6])
    else:
        ylim1, ylim2 = ylims

    print(ylim2)
    # init
    g1s = np.zeros((len(ses), len(d1s)))
    g2s = np.zeros((len(ses), len(d2s)))

    print('calc: init')

    for i, se in enumerate(ses):
        _, S1 = smatrix.one_particle(se, chlsi[0], chlso[0], d1s + offset)
        g1s[i, :] = np.abs(S1) ** 2
        g2s[i, :] = g2calc.coherent_state_tau0(se, chlsi, chlso, d2s + 2 * offset)['g2']

    print('calc: done')
    print('layout: init')

    f, (ax1, ax2) = plt.subplots(2, figsize=(6, 5))
    # AX1

    # limits
    ax1.set_xlim([np.min(2 * d1s), np.max(2 * d1s)])
    ax1.set_ylim(ylim1)

    # horizontal unit line
    ax1.semilogy(2 * d1s, np.ones(d1s.size), linewidth=1, color='#BBBBBB')
    # res
    ax1.semilogy(2 * d1s, g1s[0, :], linewidth=1.2, color='#444444')

    ax1.set_xlim([np.min(2 * d1s), np.max(2 * d1s)])
    ax1.set_ylim(ylim1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # locators
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    # labels
    ax1.set_ylabel(r'$g^{{(1)}}_{{{ins[0]},{outs[0]}}}$ (units of $f$)'.format(
        ins=[c + 1 for c in chlsi],
        outs=[c + 1 for c in chlso]))
    ax1.set_xlabel(r'$2 \delta$ (units of $\Gamma$)')

    # update font size
    update_font(ax1, 16)

    # ax2

    # limits
    ax2.set_ylim(ylim2)
    ax2.set_xlim([np.min(d2s), np.max(d2s)])

    # horizontal unit line
    ax2.semilogy(d2s, np.ones(d2s.size), linewidth=1, color='#BBBBBB')

    # linestyle
    # linestyles = ['-', (1, (9, 1.2)), (1, (9, .7, 1, .7)), (1, (9, 3))]
    linestyles = ['-', (1, (9, 1.5)), (1, (9, 1.5, 2, 1.5)), (1, (3, 1.5))]
    ls = cycle(linestyles)

    colors = plt.cm.inferno(np.linspace(.2, .8, len(ses)))
    markers = ['', 'o', '^', 's', 'o']
    markers = [''] * 8
    for i, se in enumerate(ses):
        # find label
        lbl = None if not hasattr(se, 'label') else se.label
        # ax2.semilogy(d2s, np.abs(g2s[i, :]), linewidth=1.5, ls='-', color='#222222')
        # ax2.semilogy(d2s, np.abs(g2s[i, :]), linewidth=1.2, ls='-', color='#FFFFFF')
        # ax2.semilogy(d2s, np.abs(g2s[i, :]), linewidth=0.4, color=colors[i])
        ax2.semilogy(d2s, np.abs(g2s[i, :]), label=lbl, linewidth=1.6, ls=next(ls), color=colors[i])
        # ax2.semilogy(d2s[::128], np.abs(g2s[i, ::128]), linewidth=0, color=colors[i], marker=markers[i])
        # ax2.semilogy(d2s, np.abs(g2s[i, :]), label=lbl, linewidth=1.3, ls=next(ls), color=colors[i])

    # locators
    # ax2.yaxis.set_major_locator(LogLocator(numticks=yticks))
    if yticks2 is None:
        yticks2 = [ylim2[0], 1, ylim2[1]]

    ax2.set_yticks(yticks2)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    # legend
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    lgd = None
    if hasattr(ses[0], 'label'):
        lgd = ax2.legend(loc="center left", fancybox=False, fontsize=15, edgecolor=None, title=r"$U/\Gamma$", frameon=False, bbox_to_anchor=(1, 0.5))
        # lgd = ax2.legend(loc="upper left", fancybox=False, fontsize=16, edgecolor='#222222', title=r"$U/\Gamma$")
        plt.setp(lgd.get_title(), fontsize=15)
        lgd.get_frame().set_alpha(0.0)

    # labels
    ax2.set_ylabel(r'$g^{{(2)}}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}}$'.format(
        ins=[c + 1 for c in chlsi],
        outs=[c + 1 for c in chlso]))
    ax2.set_xlabel(r'$2 \delta$ (units of \Gamma$)')

    # font size
    update_font(ax2, 16)

    plt.tight_layout()
    print('layout: done')

    print('save: init')
    plt.savefig(directory + 'g2_{ins[0]}{ins[1]}{outs[0]}{outs[1]}_coherent_tau0.pdf'.format(ins=chlsi, outs=chlso),
                bbox_inches='tight')
    plt.close(f)

    print('save: done')


def phi2_coherent(ses, chlsi, chlso, d1s, d2s, directory, offset, ylims=None, yticks=2):

    # defautls
    if ylims is None:
        ylim1 = ([1e-6, 1e6])
        ylim2 = ([1e-6, 1e6])
    else:
        ylim1, ylim2 = ylims

    # init
    g1s = np.zeros((len(ses), len(d1s)))
    g2s = np.zeros((len(ses), len(d2s)))

    print('calc: init')

    for i, se in enumerate(ses):
        _, S1 = smatrix.one_particle(se, chlsi[0], chlso[0], d1s + offset)
        g1s[i, :] = np.abs(S1) ** 2
        g2s[i, :] = g2calc.coherent_state_tau0(se, chlsi, chlso, d2s + 2 * offset)['phi2']

    print('calc: done')
    print('layout: init')

    f, (ax1, ax2) = plt.subplots(2, figsize=(6, 5))
    # AX1

    # limits
    ax1.set_xlim([np.min(d1s), np.max(d1s)])
    ax1.set_ylim(ylim1)

    # horizontal unit line
    ax1.semilogy(2 * d1s, np.ones(d1s.size), linewidth=1, color='#BBBBBB')
    # res
    ax1.semilogy(2 * d1s, g1s[0, :], linewidth=1.2, color='#444444')

    ax1.set_xlim([np.min(d1s), np.max(d1s)])
    ax1.set_ylim(ylim1)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # locators
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    # labels
    ax1.set_ylabel(r'$g^{{(1)}}_{{{ins[0]},{outs[0]}}}$'.format(
        ins=[c + 1 for c in chlsi],
        outs=[c + 1 for c in chlso]))
    ax1.set_xlabel(r'$2 \delta$ (units of $\Gamma$)')

    # update font size
    update_font(ax1, 16)

    # ax2

    # limits
    ax2.set_ylim(ylim2)
    ax2.set_xlim([np.min(d2s), np.max(d2s)])

    # horizontal unit line
    ax2.semilogy(d2s, np.ones(d2s.size), linewidth=1, color='#BBBBBB')

    # linestyle
    linestyles = ['-', (1, (9, 1.2)), (1, (9, .7, 1, .7)), (1, (9, 3))]
    ls = cycle(linestyles)
    colors = plt.cm.inferno(np.linspace(.2, .8, len(ses)))

    for i, se in enumerate(ses):
        # find label
        lbl = None if not hasattr(se, 'label') else se.label
        # ax2.semilogy(d2s, np.abs(g2s[i, :]), linewidth=1.5, ls='-', color='#222222')
        # ax2.semilogy(d2s, np.abs(g2s[i, :]), linewidth=1.2, ls='-', color='#FFFFFF')
        ax2.semilogy(d2s, np.abs(g2s[i, :]), linewidth=0.4, color=colors[i])
        ax2.semilogy(d2s, np.abs(g2s[i, :]), label=lbl, linewidth=1.2, ls=next(ls), color=colors[i])

    # locators
    # ax2.yaxis.set_major_locator(LogLocator(numticks=yticks))
    ax2.set_yticks([ylim2[0], 1, ylim2[1]])
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    # legend
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    lgd = None
    if hasattr(ses[0], 'label'):
        lgd = ax2.legend(loc="center left", fancybox=False, fontsize=15, edgecolor=None, title=r"$U/\Gamma$", frameon=False, bbox_to_anchor=(1, 0.5))
        # lgd = ax2.legend(loc="upper left", fancybox=False, fontsize=16, edgecolor='#222222', title=r"$U/\Gamma$")
        plt.setp(lgd.get_title(), fontsize=15)
        lgd.get_frame().set_alpha(0.0)

    # labels
    ax2.set_ylabel(r'$\phi^{{(2)}}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}}$'.format(
        ins=[c + 1 for c in chlsi],
        outs=[c + 1 for c in chlso]))
    ax2.set_xlabel(r'$2 \delta$ (units of \Gamma$)')

    # font size
    update_font(ax2, 16)

    plt.tight_layout()
    print('layout: done')

    print('save: init')
    plt.savefig(directory + 'phi2_{ins[0]}{ins[1]}{outs[0]}{outs[1]}_coherent_tau0.pdf'.format(ins=chlsi, outs=chlso),
                bbox_inches='tight')
    plt.close(f)

    print('save: done')


def g2_coherent_tau(se, chlsi, chlso, Es, taus, directory="", logscale=False, ticks=None, title=None, verbose=False):
    """
    Plots intensity-intensity correlation g2 for weakly coherent state
    as a function of times taus and two-photon energies.

    Parameters
    ----------
    se : scattering.Setup object
        The current scattering setup
    chlsi : tuple
        The two channels of incoming photons.
    chlso : tuple
        The two channels of outgoing photons.
    Es : array-like
        List of two-photon energies
    taus : array-like
        List of times/position differences
    directory : string
        Directory for saving the view.
    """
    g2s = np.zeros((len(Es), len(taus)))
    for i, E in enumerate(Es):
        g2s[i, :] = g2calc.coherent_state(se, chlsi, chlso, E, taus, verbose=verbose)['g2']
    # g2
    fig = plt.figure(figsize=(7.75, 4))
    extra = ''

    if title is None:
        title = ''

    if ticks is None:
        ticks = [0.6, 0.8, 1.0, 1.2, 1.4]

    if len(Es) > 15 and len(taus) > 15:
        # create a nice colored 2d view

        if logscale is False:
            plt.pcolormesh(Es, taus, g2s.T, cmap='RdBu_r', rasterized=True,
                           vmin=ticks[0],
                           vmax=ticks[-1])
            cb = plt.colorbar(ticks=ticks)
            ticksom = [r'${}$'.format(i) for i in ticks]
            cb.ax.set_yticklabels(ticksom)

            # im = plt.imshow(Z, interpolation='bilinear', origin='lower',
            # cmap=cm.gray, extent=(-3, 3, -2, 2))
            levels = ticks
            CS = plt.contour(g2s.T, levels,
                             linestyles=':',
                             colors='k',
                             origin='lower',
                             linewidths=.5,
                             extent=(Es[0], Es[-1], taus[0], taus[-1])

                             )

        else:
            ax = plt.pcolormesh(Es, taus, g2s.T, cmap='RdBu_r', rasterized=True,
                                norm=LogNorm(vmin=ticks[0], vmax=ticks[-1]))
            cb = plt.colorbar(ticks=ticks)

            levels = ticks
            CS = plt.contour(g2s.T, levels,
                             linestyles=':',
                             colors='k',
                             origin='lower',
                             linewidths=.5,
                             extent=(Es[0], Es[-1], taus[0], taus[-1])
                             )

        for t in cb.ax.get_yticklabels():
            t.set_fontsize(18)

        locx = ticker.MultipleLocator(base=(Es[-1] - Es[0]) / 5)  # this locator puts ticks at regular intervals
        locy = ticker.MultipleLocator(base=taus[-1] / 4)  # this locator puts ticks at regular intervals
        ax = plt.gca()
        ax.xaxis.set_major_locator(locx)
        ax.yaxis.set_major_locator(locy)

        plt.xlabel(r'$2 \delta$ (units of $\Gamma$)')
        plt.xlim([np.min(Es), np.max(Es)])
        plt.ylabel(r'$\tau$')
        plt.ylim([np.min(taus), np.max(taus)])

        # plt.title(r'$\log_{{10}} g^{{(2)}}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}}$'.format(ins=chlsi, outs=chlso))
        plt.title(r'$g^{{(2)}}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}} (\tau)$ {title}'.format(ins=[c + 1 for c in chlsi], outs=[c + 1 for c in chlso], title=title))
    elif len(Es) < 15:
        extra = '_E{Es}'.format(Es=Es)
        # create some nice graphs to look at
        g2max = np.max(g2s)
        for i, E in enumerate(Es):
            if logscale:
                plt.semilogy(taus, g2s[i, :], label=r'$E={0}$'.format(E))
            else:
                plt.plot(taus, g2s[i, :], label=r'$E={0}$'.format(E))

        plt.xlabel(r'$\tau$')
        plt.xlim([np.min(taus), np.max(taus)])
        # plt.ylabel(r'$\log_{{10}} g^{{(2)}}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}}$'.format(ins=chlsi, outs=chlso))
        plt.title(r'$g^{{(2)}}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}} (\tau)$ {title}'.format(ins=[c + 1 for c in chlsi], outs=[c + 1 for c in chlso], title=title))

        plt.legend()
    elif len(taus) < 15:
        extra = '_s{taus}'.format(taus=taus)
        # create some nice graphs to look at
        g2max = np.max(g2s)
        for i, tau in enumerate(taus):
            if logscale:
                plt.semilogy(Es, g2s[:, i], label=r'$\tau={0}$'.format(tau))
            else:
                plt.plot(Es, g2s[:, i], label=r'$\tau={0}$'.format(tau))

        plt.xlabel(r'$\tau$')
        plt.xlim([np.min(Es), np.max(Es)])
        plt.ylabel(r'$g^{{(2)}}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}}$'.format(ins=[c + 1 for c in chlsi], outs=[c + 1 for c in chlso]))
        plt.legend()

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

    box = cb.ax.get_position()
    cb.ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

    update_font(plt.gca(), 18)
    # plt.tight_layout()
    plt.savefig(directory + 'g2_{ins[0]}{ins[1]}{outs[0]}{outs[1]}_coherent_tau{extra}.pdf'.format(ins=chlsi, outs=chlso, extra=extra))


def update_font(ax, fs):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fs)
