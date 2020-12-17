# import os
# import sys
# print(os.getcwd())
import time_to_orbit as tto
# #os.chdir('/Users/Riley/PycharmProjects/mms_data/mms_sitl_ground_loop');
# repo_dir = os.getcwd()
# #if not os.path.isfile(os.path.join(repo_dir, 'util.py')):
# #    raise ValueError('Could not automatically determine the model root.')
# #if repo_dir not in sys.path:
# #    sys.path.append(repo_dir)
#
#
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
import io
import re
import requests
import csv
import pymms
from tqdm import tqdm
from cdflib import epochs
from urllib.parse import parse_qs
import urllib3
import warnings
from scipy.io import readsav
from getpass import getpass
import cdflib
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pathlib
outdir = None
# tai_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
#
#
#
# # the following is the code from notebook figure3_table3.ipynb
# #
# if outdir is not None:
#     outdir = pathlib.Path(outdir).expanduser().absolute()
#
#
#
#
# sc = 'mms1'
# i_clim = (3, 18.5)
# e_clim = (12, 21.5)
# b_lim = (-40, 75)
# n_lim = (5e-1, 2e1)
# sel_lim = (0, 200)
#
#
# t0 = dt.datetime(2019, 10, 22, 19, 0, 0)
# t1 = dt.datetime(2019, 10, 22, 23, 10, 0)
# delta = 10
#
# orbit = tto.time_to_orbit(t0,sc,delta)
# sroi = 1
#
#
# #'sitl+back' option includes selections submitted to the back structure
# #'mp-dl-unh' is the name of the GLS model
# sitl_data = tto.selections('sitl+back', t0, t1)
# abs_data = tto.selections('abs', t0, t1)
# gls_data = tto.selections('mp-dl-unh', t0, t1)
#
# print('|------------------------------------------------------------------------')
# print('| Selections made on orbit {0} between {1} and {2}'.format(orbit, t0, t1))
# print('|--------------------------|')
# print('| SITL and Back Structure: |')
# print('|--------------------------|')
# tto.print_segments(sitl_data)
# print('|------|')
# print('| ABS: |')
# print('|------|')
# tto.print_segments(abs_data)
# print('|------|')
# print('| GLS: |')
# print('|------|')
# tto.print_segments(gls_data)
#
#
#
# # Save the selections
# if outdir is not None:
#     tto.write_csv(outdir / 'table3_orbit-{0}_sroi-{1}.csv'.format(orbit, sroi),
#                   [*sitl_data, *abs_data, *gls_data])
#
#
# # Plot the magnetopause interval
# fig, axes = tto.plot_burst_selections(sc, t0, t1, figsize=(2.8,5))
# axes[0][0].images[0].set_clim(i_clim)
# axes[0][0].set_title('{0} Orbit {1} SROI{2}'.format(sc.upper(), orbit, sroi))
# axes[1][0].images[0].set_clim(e_clim)
# axes[2][0].set_ylim(b_lim)
# axes[3][0].set_ylim(n_lim)
# for i in range(3):
#     axes[i+4][0].set_ylim(sel_lim)
#
# # Save the figure
# if outdir is not None:
#     plt.savefig(outdir / 'figure3_orbit-{0}_sroi-{1}.png'.format(orbit, sroi))
# plt.show()
#
#
#
#
#
# #second figure
#
# t0 = dt.datetime(2019, 12, 17, 18, 15, 0)
# t1 = dt.datetime(2019, 12, 17, 18, 55, 0)
# orbit = tto.time_to_orbit(t0,sc,delta)
# sroi = 1
#
#
# # 'sitl+back' option includes selections submitted to the back structure
# # 'mp-dl-unh' is the name of the GLS model
# sitl_data = tto.selections('sitl+back', t0, t1)
# abs_data = tto.selections('abs', t0, t1)
# gls_data = tto.selections('mp-dl-unh', t0, t1)
#
# print('|------------------------------------------------------------------------')
# print('| Selections made on orbit {0} between {1} and {2}'.format(orbit, t0, t1))
# print('|--------------------------|')
# print('| SITL and Back Structure: |')
# print('|--------------------------|')
# tto.print_segments(sitl_data)
# print('|------|')
# print('| ABS: |')
# print('|------|')
# tto.print_segments(abs_data)
# print('|------|')
# print('| GLS: |')
# print('|------|')
# tto.print_segments(gls_data)
#
#
# # Save the selections
# if outdir is not None:
#     tto.write_csv(outdir / 'table3_orbit-{0}_sroi-{1}.csv'.format(orbit, sroi),
#                   [*sitl_data, *abs_data, *gls_data])
#
#
# # Plot the magnetopause interval on orbit 1062
#
# fig, axes = tto.plot_burst_selections(sc, t0, t1, figsize=(2.8,5))
# axes[0][0].images[0].set_clim(i_clim)
# axes[0][0].set_title('{0} Orbit {1} SROI{2}'.format(sc.upper(), orbit, sroi))
# axes[1][0].images[0].set_clim(e_clim)
# axes[2][0].set_ylim(b_lim)
# axes[3][0].set_ylim(n_lim)
# for i in range(3):
#     axes[i+4][0].set_ylim(sel_lim)
#
# # Save the figure
# if outdir is not None:
#     plt.savefig(outdir / 'figure3_orbit-{0}_sroi-{1}.png'.format(orbit, sroi))
# plt.show()
#
#
# # third figure
#
#
# t0 = dt.datetime(2020, 1, 17, 19, 30, 0)
# t1 = dt.datetime(2020, 1, 17, 21, 0, 0)
# orbit = tto.time_to_orbit(t0)
# sroi = 3
#
#
# # 'sitl+back' option includes selections submitted to the back structure
# # 'mp-dl-unh' is the name of the GLS model
# sitl_data = tto.selections('sitl+back', t0, t1)
# abs_data = tto.selections('abs', t0, t1)
# gls_data = tto.selections('mp-dl-unh', t0, t1)
#
# print('|------------------------------------------------------------------------')
# print('| Selections made on orbit {0} between {1} and {2}'.format(orbit, t0, t1))
# print('|--------------------------|')
# print('| SITL and Back Structure: |')
# print('|--------------------------|')
# tto.print_segments(sitl_data)
# print('|------|')
# print('| ABS: |')
# print('|------|')
# tto.print_segments(abs_data)
# print('|------|')
# print('| GLS: |')
# print('|------|')
# tto.print_segments(gls_data)
#
#
# # Save the selections
# if outdir is not None:
#     tto.write_csv(outdir / 'table3_orbit-{0}_sroi-{1}.csv'.format(orbit, sroi),
#                   [*sitl_data, *abs_data, *gls_data])
#
#
#
# # Plot the magnetopause interval on orbit 1075
# fig, axes = tto.plot_burst_selections(sc, t0, t1, figsize=(2.8,5))
# axes[0][0].images[0].set_clim(i_clim)
# axes[0][0].set_title('{0} Orbit {1} SROI{2}'.format(sc.upper(), orbit, sroi))
# axes[1][0].images[0].set_clim(e_clim)
# axes[2][0].set_ylim(b_lim)
# axes[3][0].set_ylim(n_lim)
# for i in range(3):
#      axes[i+4][0].set_ylim(sel_lim)
#
# if outdir is not None:
#     plt.savefig(outdir / 'figure3_orbit-{0}_sroi-{1}.png'.format(orbit, sroi))
# plt.show()





# new code from argalls github
# import pymms
# outdir = pymms.config['gls_root'] + '/gls_paper'
# fig_type = 'png' # png, jpg, svg, eps, ...
#
# import sys
# repo_dir = os.getcwd()
# if not os.path.isfile(os.path.join(repo_dir, 'util.py')):
#     raise ValueError('Could not automatically determine the model root.')
# if repo_dir not in sys.path:
#     sys.path.append(repo_dir)
# import util
#
# from pymms.sdc import selections as sel
# from pymms.sdc import mrmms_sdc_api as api
import datetime as dt
from matplotlib import pyplot as plt
import pathlib

# if outdir is not None:
#     outdir = pathlib.Path(outdir).expanduser().absolute()




# t0 = dt.datetime(2015, 10, 30, 5, 14, 0)
# t1 = dt.datetime(2015, 10, 30, 5, 18, 0)
# t0 = dt.datetime(2019, 10, 4, 1, 0, 53)
# t1 = dt.datetime(2019, 10, 4, 23, 0, 33)
# t0 = dt.datetime(2020, 1, 17, 19, 50, 0)
# t1 = dt.datetime(2020, 1, 17, 20, 30, 0)
# orbit = tto.time_to_orbit(t0)
# sroi = 3
sc = 'mms1'
i_clim = (3, 18.5)
e_clim = (12, 21.5)
b_lim = (-40, 75)
n_lim = (5e-1, 1e2)
v_lim = (-175,175)
sel_lim = (0, 200)
t0 = dt.datetime(2016, 2, 1, 0, 0, 0)
t1 = dt.datetime(2016, 3, 29, 23, 45, 0)

orbit = tto.time_to_orbit(t0)
sroi = 1

# 'sitl+back' option includes selections submitted to the back structure
# 'mp-dl-unh' is the name of the GLS model
sitl_data = tto.selections('sitl+back', t0, t1, combine=True, sort=True, filter = 'MP with jet')
abs_data = tto.selections('abs', t0, t1, combine=True, sort=True, filter = 'MP with jet')
gls_data = tto.selections('mp-dl-unh', t0, t1, combine=True, sort=True, filter = 'MP with jet')

print('|------------------------------------------------------------------------')
print('| Selections made on orbit {0} between {1} and {2}'.format(orbit, t0, t1))
print('|--------------------------|')
print('| SITL and Back Structure: |')
print('|--------------------------|')
tto.print_segments(sitl_data)
print('|------|')
print('| ABS: |')
print('|------|')
tto.print_segments(abs_data)
print('|------|')
print('| GLS: |')
print('|------|')
tto.print_segments(gls_data)


# Plot the magnetopause interval
# fig, axes = tto.plot_burst_selections(sc, t0, t1, figsize=(2.8,5))
# axes[0][0].images[0].set_clim(i_clim)
# axes[0][0].set_title('{0} Orbit {1} SROI{2}'.format(sc.upper(), orbit, sroi))
# axes[1][0].images[0].set_clim(e_clim)
# axes[2][0].set_ylim(b_lim)
# axes[3][0].set_ylim(n_lim)
# axes[4][0].set_ylim(v_lim)
# for i in range(4):
#     axes[i+4][0].set_ylim(sel_lim)
#
#
# plt.show()


# t0 = dt.datetime(2019, 11, 16, 7, 15, 0)
# t1 = dt.datetime(2019, 11, 16, 8, 45, 0)
# orbit = tto.time_to_orbit(t0)
# sroi = 1
#
# # 'sitl+back' option includes selections submitted to the back structure
# # 'mp-dl-unh' is the name of the GLS model
# sitl_data = tto.selections('sitl+back', t0, t1, combine=True, sort=True)
# abs_data = tto.selections('abs', t0, t1, combine=True, sort=True)
# gls_data = tto.selections('mp-dl-unh', t0, t1, combine=True, sort=True)
#
# print('|------------------------------------------------------------------------')
# print('| Selections made on orbit {0} between {1} and {2}'.format(orbit, t0, t1))
# print('|--------------------------|')
# #print('| SITL and Back Structure: |')
# print('|--------------------------|')
# tto.print_segments(sitl_data)
# print('|------|')
# print('| ABS: |')
# print('|------|')
# tto.print_segments(abs_data)
# print('|------|')
# print('| GLS: |')
# print('|------|')
# tto.print_segments(gls_data)
# #
# # # Save the selections
# # # if outdir is not None:
# # #     tto.write_csv(outdir / 'table_case-studies_orbit-{0}_sroi-{1}.csv'.format(orbit, sroi),
# # #                   [*sitl_data, *abs_data, *gls_data])
# #
# # # Plot the magnetopause interval on orbit 1062
# fig, axes = tto.plot_burst_selections(sc, t0, t1, figsize=(2.8,5))
# axes[0][0].images[0].set_clim(i_clim)
# axes[0][0].set_title('{0} Orbit {1} SROI{2}'.format(sc.upper(), orbit, sroi))
# axes[1][0].images[0].set_clim(e_clim)
# axes[2][0].set_ylim(b_lim)
# axes[3][0].set_ylim(n_lim)
# for i in range(3):
#     axes[i+4][0].set_ylim(sel_lim)
#
# # Save the figure
# # if outdir is not None:
# #     plt.savefig(outdir / 'figure_case-studies_orbit-{0}_sroi-{1}.{2}'.format(orbit, sroi, fig_type),
# #                 dpi=300, transparent=True)
# plt.show()
#
# t0 = dt.datetime(2020, 1, 17, 19, 50, 0)
# t1 = dt.datetime(2020, 1, 17, 20, 30, 0)
# orbit = tto.time_to_orbit(t0)
# sroi = 3
#
# # 'sitl+back' option includes selections submitted to the back structure
# # 'mp-dl-unh' is the name of the GLS model
# sitl_data = tto.selections('sitl+back', t0, t1, combine=True, sort=True)
# abs_data = tto.selections('abs', t0, t1, combine=True, sort=True)
# gls_data = tto.selections('mp-dl-unh', t0, t1, combine=True, sort=True)
#
# print('|------------------------------------------------------------------------')
# print('| Selections made on orbit {0} between {1} and {2}'.format(orbit, t0, t1))
# print('|--------------------------|')
# print('| SITL and Back Structure: |')
# print('|--------------------------|')
# tto.print_segments(sitl_data)
# print('|------|')
# print('| ABS: |')
# print('|------|')
# tto.print_segments(abs_data)
# print('|------|')
# print('| GLS: |')
# print('|------|')
# tto.print_segments(gls_data)
#
#
# # Save the selections
# # if outdir is not None:
# #     tto.write_csv(outdir / 'table_case-studies_orbit-{0}_sroi-{1}.csv'.format(orbit, sroi),
# #                   [*sitl_data, *abs_data, *gls_data])
#
# # Plot the magnetopause interval on orbit 1075
# fig, axes = tto.plot_burst_selections(sc, t0, t1, figsize=(2.8,5))
# axes[0][0].images[0].set_clim(i_clim)
# axes[0][0].set_title('{0} Orbit {1} SROI{2}'.format(sc.upper(), orbit, sroi))
# axes[1][0].images[0].set_clim(e_clim)
# axes[2][0].set_ylim(b_lim)
# axes[3][0].set_ylim(n_lim)
# for i in range(3):
#     axes[i+4][0].set_ylim(sel_lim)
#
# # if outdir is not None:
# #     plt.savefig(outdir / 'figure_case-studies_orbit-{0}_sroi-{1}.{2}'.format(orbit, sroi, fig_type),
# #                 dpi=300, transparent=True)
# plt.show()











#
# start_date = dt.datetime(2019,10,26,6,55,0)
# end_date = dt.datetime(2019,10,26,7,3,0)
# orbit = tto.time_to_orbit(start_date)
# sroi = 1
# mode = 'srvy'
# level = 'l2'
#
# # magnetic field data
# b_vname = '_'.join((sc, 'fgm', 'b', 'gse', mode, level))
# mms = tto.MrMMS_SDC_API(sc, 'fgm', mode, level,
#                     start_date=start_date, end_date=end_date)
# files = mms.download_files()
# files = tto.sort_files(files)[0]
#
# fgm_data = tto.from_cdflib(files, b_vname,
#                        start_date, end_date)
#
# # Ion energy and number density
# fpi_mode = 'fast'
# ni_vname = '_'.join((sc, 'dis', 'numberdensity', fpi_mode))
# espec_i_vname = '_'.join((sc, 'dis', 'energyspectr', 'omni', fpi_mode))
# mms = tto.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
#                     optdesc='dis-moms',
#                     start_date=start_date, end_date=end_date)
# files = mms.download_files()
# files = tto.sort_files(files)[0]
#
# ni_data = tto.from_cdflib(files, ni_vname,
#                       start_date, end_date)
# especi_data = tto.from_cdflib(files, espec_i_vname,
#                           start_date, end_date)
#
# # electron energy
# ne_vname = '_'.join((sc, 'des', 'numberdensity', fpi_mode))
# espec_e_vname = '_'.join((sc, 'des', 'energyspectr', 'omni', fpi_mode))
# mms =tto. MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
#                     optdesc='des-moms',
#                     start_date=start_date, end_date=end_date)
# files = mms.download_files()
# files = tto.sort_files(files)[0]
# espece_data = tto.from_cdflib(files, espec_e_vname,
#                           start_date, end_date)
#
# print(fgm_data)