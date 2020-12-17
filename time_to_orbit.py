import os
import sys
import time_to_orbit as tto
#os.chdir('/Users/Riley/PycharmProjects/mms_data/mms_sitl_ground_loop');
repo_dir = os.getcwd()
#if not os.path.isfile(os.path.join(repo_dir, 'util.py')):
#    raise ValueError('Could not automatically determine the model root.')
#if repo_dir not in sys.path:
#    sys.path.append(repo_dir)
#import util
#from pymms.sdc import selections as sel
#from pymms.sdc import mrmms_sdc_api as api
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
import io
import re
import requests
import csv
# import pymms
# from tqdm import tqdm
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
import function_folder as ff
outdir = None

tai_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])


class BurstSegment:
    def __init__(self, tstart, tstop, fom, discussion,
                 sourceid=None, createtime=None):
        '''
        Create an object representing a burst data segment.

        Parameters
        ----------
        fom : int, float
            Figure of merit given to the selection
        tstart : int, str, `datetime.datetime`
            The start time of the burst segment, given as a `datetime` object,
            a TAI time in seconds since 1 Jan. 1958, or as a string formatted
            as `yyyy-MM-dd hh:mm:SS`. Times are converted to `datetimes`.
        tstop : int, str, `datetime.datetime`
            The stop time of the burst segment, similar to `tstart`.
        discussion : str
            Description of the segment provided by the SITL
        sourceid : str
            Username of the SITL that made the selection
        file : str
            Name of the file containing the selection
        '''

        # Convert tstart to datetime
        if isinstance(tstart, str):
            tstart = dt.datetime.strptime(tstart, '%Y-%m-%d %H:%M:%S')
        elif isinstance(tstart, int):
            tstart = self.__class__.tai_to_datetime(tstart)

        # Convert tstop to datetime
        if isinstance(tstop, str):
            tstop = dt.datetime.strptime(tstop, '%Y-%m-%d %H:%M:%S')
        elif isinstance(tstop, int):
            tstop = self.__class__.tai_to_datetime(tstop)

        self.discussion = discussion
        self.createtime = createtime
        self.fom = fom
        self.sourceid = sourceid
        self.tstart = tstart
        self.tstop = tstop

    def __str__(self):
        return ('{0}   {1}   {2:3.0f}   {3}'
                .format(self.tstart, self.tstop, self.fom, self.discussion)
                )

    def __repr__(self):
        return ('selections.BurstSegment({0}, {1}, {2:3.0f}, {3})'
                .format(self.tstart, self.tstop, self.fom, self.discussion)
                )

    @staticmethod
    def datetime_to_list(t):
        return [t.year, t.month, t.day,
                t.hour, t.minute, t.second,
                t.microsecond // 1000, t.microsecond % 1000, 0
                ]

    @classmethod
    def datetime_to_tai(cls, t):
        t_list = cls.datetime_to_list(t)
        return int((epochs.CDFepoch.compute_tt2000(t_list) - tai_1958) // 1e9)

    @classmethod
    def tai_to_datetime(cls, t):
        tepoch = epochs.CDFepoch()
        return tepoch.to_datetime(t * int(1e9) + tai_1958)

    @property
    def start_time(self):
        return self.tstart.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def stop_time(self):
        return self.tstop.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def taistarttime(self):
        return self.__class__.datetime_to_tai(self.tstart)

    @property
    def taiendtime(self):
        return self.__class__.datetime_to_tai(self.tstop)


def _burst_data_segments_to_burst_segment(data):
    '''
    Turn selections created by `MrMMS_SDC_API.burst_data_segements` and turn
    them into `BurstSegment` instances.

    Parameters
    ----------
    data : dict
        Data associated with each burst segment

    Returns
    -------
    result : list of `BurstSegment`
        Data converted to `BurstSegment` instances
    '''
    # Look at createtime and finishtime keys to see if either can
    # substitute for a file name time stamp

    result = []
    for tstart, tend, fom, discussion, sourceid, createtime, status in \
            zip(data['tstart'], data['tstop'], data['fom'],
                data['discussion'], data['sourceid'], data['createtime'],
                data['status']
                ):
        segment = BurstSegment(tstart, tend, fom, discussion,
                               sourceid=sourceid,
                               createtime=createtime)
        segment.status = status
        result.append(segment)
    return result


def _get_selections(type, start, stop,
                    sort=False, combine=False, latest=True, unique=False,
                    metadata=False, filter=None, case_sensitive=False):
    if latest and unique:
        raise ValueError('latest and unique keywords '
                         'are mutually exclusive.')

    # Burst selections can be made multiple times within the orbit.
    # Multiple submissions will be appear as repeated entires with
    # different create times, but only the last submission is the
    # official submission. To ensure that the last submission is
    # returned, look for selections submitted through the following
    # orbit as well.
    orbit_start = time_to_orbit(start)
    orbit_stop = time_to_orbit(stop) + 1

    # Get the selections
    data = burst_selections(type, orbit_start, orbit_stop)

    # Turn the data into BurstSegments. Adjacent segments returned
    # by `sdc.sitl_selections` have a 0 second gap between stop and
    # start times. Those returned by `sdc.burst_data_segments` are
    # separated by 10 seconds.
    if type in ('abs', 'sitl', 'gls', 'mp-dl-unh'):
        delta_t = 0.0
        converter = _sitl_selections_to_burst_segment
    elif type == 'sitl+back':
        delta_t = 10.0
        converter = _burst_data_segments_to_burst_segment
    else:
        raise ValueError('Invalid selections type {}'.format(type))

    # Convert data into BurstSegments
    # If there were no selections made, data will be empty without
    # keys (and throws error in _sitl_selections_to_burst_segment).
    try:
        data = converter(data)
    except KeyError:
        return []

    # Get metadata associated with orbit, sroi, and metadata
    if metadata:
        _get_segment_data(data, orbit_start, orbit_stop)

    # The official selections are those from the last submission
    # containing selections within the science_roi. Get rid of
    # all submissions except the last.
    if latest:
        data = _latest_segments(data, orbit_start, orbit_stop,
                                sitl=(type == 'sitl+back'))

    # Get rid of extra selections obtained by changing the
    # start and stop interval
    data = [segment
            for segment in data
            if (segment.tstart >= start) \
            and (segment.tstop <= stop)]

    # Additional handling of data
    if combine:
        combine_segments(data, delta_t)
    if sort:
        data = sort_segments(data)
    if unique:
        data = remove_duplicate_segments(data)
    if filter is not None:
        data = filter_segments(data, filter,
                               case_sensitive=case_sensitive)

    return data


def _get_segment_data(data, orbit_start, orbit_stop, sc='mms1'):
    '''
    Add metadata associated with the orbit, SROI, and SITL window
    to each burst segment.

    Parameters
    ----------
    data : list of BurstSegments
        Burst selections. Selections are altered in-place to
        have new attributes:
        Attribute            Description
        ==================   ======================
        orbit                Orbit number
        orbit_tstart         Orbit start time
        orbit_tstop          Orbit stop time
        sroi                 SROI number
        sroi_tstart          SROI start time
        sroi_tstop           SROI stop time
        sitl_window_tstart   SITL window start time
        sitl_window_tstop    SITL window stop time
        ==================   ======================
    orbit_start, orbit_stop : int or np.integer
        Orbit interval in which selections were made
    sc : str
        Spacecraft identifier
    '''
    idx = 0
    for iorbit in range(orbit_start, orbit_stop + 1):
        # The start and end times of the sub-regions of interest.
        # These are the times in which selections can be made for
        # any given orbit.
        orbit = mission_events('orbit', iorbit, iorbit, sc=sc)
        sroi = mission_events('sroi', iorbit, iorbit, sc=sc)
        window = mission_events('sitl_window', iorbit, iorbit)
        tstart = min(sroi['tstart'])
        tend = max(sroi['tend'])

        # Find the burst segments that were selected within the
        # current SROI

        while idx < len(data):
            segment = data[idx]

            # Filter out selections from the previous orbit(s)
            # Stop when we get to the next orbit
            if segment.tstop < tstart:
                idx += 1
                continue
            if segment.tstart > tend:
                break

            # Keep segments from the same submission (create time).
            # If there is a new submission within the orbit, take
            # selections from the new submission and discard those
            # from the old.
            segment.orbit = iorbit
            segment.orbit_tstart = orbit['tstart'][0]
            segment.orbit_tstop = orbit['tend'][0]

            sroi_data = _get_sroi_number(sroi, segment.tstart, segment.tstop)
            segment.sroi = sroi_data[0]
            segment.sroi_tstart = sroi_data[1]
            segment.sroi_tstop = sroi_data[2]

            # The SITL Window is not defined as of orbit 1098, when
            # a SITL-defined window was implemented
            if iorbit < 1098:
                segment.sitl_window_tstart = window['tstart']
                segment.sitl_window_tstop = window['tend']

            idx += 1


def _get_sroi_number(sroi, tstart, tstop):
    '''
    Determine which sub-region of interest (SROI) in which a given
    time interval resides.

    Parameters
    ----------
    sroi : dict
        SROI information for a specific orbit
    tstart, tstop : `datetime.datetime`
        Time interval

    Returns
    -------
    result : tuple
       The SROI number and start and stop times.
    '''
    sroi_num = 0
    for sroi_tstart, sroi_tstop in zip(sroi['tstart'], sroi['tend']):
        sroi_num += 1
        if (tstart >= sroi_tstart) and (tstop <= sroi_tstop):
            break

    return sroi_num, sroi_tstart, sroi_tstop


def _latest_segments(data, orbit_start, orbit_stop, sitl=False):
    '''
    Return the latest burst selections submission from each orbit.

    Burst selections can be submitted multiple times but only
    the latest file serves as the official selections file.

    For the SITL, the SITL makes selections within the SITL
    window. If data is downlinked after the SITL window closes,
    selections on that data are submitted separately into the
    back structure and should be appended to the last submission
    that took place within the SITL window.

    Parameters
    ----------
    data : list of BurstSegments
        Burst segment selections
    orbit_start, orbit_stop : int or np.integer
        Orbits in which selections were made
    sitl : bool
        If true, the burst selections were made by the SITL and
        there are back structure submissions to take into
        consideration

    Returns
    -------
    results : list of BurstSegments
        The latest burst selections
    '''
    result = []
    for orbit in range(orbit_start, orbit_stop + 1):
        # The start and end times of the sub-regions of interest.
        # These are the times in which selections can be made for
        # any given orbit.

        # SROI information is not available before orbit 239.
        # The first SROI is defined on 2015-11-06, which is only a
        # couple of months after formal science data collection began
        # on 2015-09-01.

        # However, similar information is available starting at
        # 2015-08-10 (orbit 151) in the science_roi event type. It's
        # nearly equivalent because there was only a single SROI per
        # orbit during the earlier mission phases, but science_roi is
        # the span across all spacecraft.
        if orbit < 239:
            sroi = mission_events('science_roi', orbit, orbit)
        else:
            sroi = mission_events('sroi', orbit, orbit)
        tstart = min(sroi['tstart'])
        tend = max(sroi['tend'])

        # The SITL Window is not defined as of orbit 1098, when
        # a SITL-defined window was implemented
        if orbit >= 1098:
            sitl = False

        # Need to know when the SITL window closes in order to
        # keep submissions to the back structure.
        if sitl:
            sitl_window = mission_events('sitl_window', orbit, orbit)

        # Find the burst segments that were selected within the
        # current SROI
        create_time = None
        orbit_segments = []
        for idx, segment in enumerate(data):
            # Filter out selections from the previous orbit(s)
            # Stop when we get to the next orbit
            if segment.tstop < tstart:
                continue
            if segment.tstart > tend:
                break

            # Initialize with the first segment within the orbit.
            # Create times are the same for all selections within
            # a single submission. There may be several submissions
            # per orbit.
            if create_time is None:
                create_time = segment.createtime

            # Keep segments from the same submission (create time).
            # If there is a new submission within the orbit, take
            # selections from the new submission and discard those
            # from the old.
            #
            # Submissions to the back structure occur after the
            # SITL window has closed and are in addition to whatever
            # the latest submission was.
            #
            # GLS and ABS selections can occur after the SITL window
            # closes, but those are treated the same as selections
            # made within the SITL window.
            if abs(create_time - segment.createtime) < dt.timedelta(seconds=10):
                orbit_segments.append(segment)
            elif segment.createtime > create_time:
                if sitl and (segment.createtime > sitl_window['tend'][0]):
                    orbit_segments.append(segment)
                else:
                    create_time = segment.createtime
                    orbit_segments = [segment]
            else:
                continue

        # Truncate the segments and append this orbit's submissions
        # to the overall results.
        data = data[idx:]
        result.extend(orbit_segments)

    return result


def _mission_events_to_burst_segment(data):
    '''
    Turn selections created by `MrMMS_SDC_API.mission_events` and turn
    them into `BurstSegment` instances.

    Parameters
    ----------
    data : dict
        Data associated with each burst segment

    Returns
    -------
    result : list of `BurstSegment`
        Data converted to `BurstSegment` instances
    '''
    raise NotImplementedError


def _sitl_selections_to_burst_segment(data):
    '''
    Turn selections created by `MrMMS_SDC_API.sitl_selections` and turn
    them into `BurstSegment` instances.

    Parameters
    ----------
    data : dict
        Data associated with each burst segment

    Returns
    -------
    result : list of `BurstSegment`
        Data converted to `BurstSegment` instances
    '''
    result = []
    for idx in range(len(data['fom'])):
        result.append(BurstSegment(data['tstart'][idx], data['tstop'][idx],
                                   data['fom'][idx], data['discussion'][idx],
                                   sourceid=data['sourceid'][idx],
                                   createtime=data['createtime'][idx],
                                   )
                      )
    return result


def combine_segments(data, dt_contig=0):
    '''
    Combine contiguous burst selections into single selections.

    Parameters
    ----------
    data : list of `BurstSegment`
        Selections to be combined.
    dt_contig : int
        Time interval between adjacent selections. For selections
        returned by `pymms.sitl_selections()`, this is 0. For selections
        returned by `pymms.burst_data_segment()`, this is 10.
    '''
    # Any time delta > dt_contig sec indicates the end of a contiguous interval
    t_deltas = [(seg1.tstart - seg0.tstop).total_seconds()
                for seg1, seg0 in zip(data[1:], data[:-1])
                ]

    # Time deltas has one fewer element than data at this stage. Append
    # infinity to the time deltas to indicate that the last element in data
    # does not have a contiguous neighbor. This will make the number of
    # elements in each array equal UNLESS data is empty. Do not append if
    # data is empty (to avoid indexing errors below).
    if len(data) > 0:
        t_deltas.append(1000)

    icontig = 0  # Current contiguous interval
    result = []

    # Check if adjacent elements are continuous in time
    #   - Use itertools.islice to select start index without copying array
    for idx, t_delta in enumerate(t_deltas):
        # Contiguous segments are separated by dt_contig seconds.
        # The last element of t_delta = 1000, so not pass this
        # condition and idx+1 will not cause an IndexError
        if t_delta == dt_contig:
            # And unique segments have the same fom and discussion
            if (data[icontig].fom == data[idx + 1].fom) and \
                    (data[icontig].discussion == data[idx + 1].discussion):
                continue

        # End of a contiguous interval
        data[icontig].tstop = data[idx].tstop

        # Next interval
        icontig = icontig + 1

        # Move data for new contiguous segments to beginning of array
        try:
            data[icontig] = data[idx + 1]
        except IndexError:
            pass

    # Truncate data beyond last contiguous interval
    del data[icontig:]


def filter_segments(data, filter, case_sensitive=False):
    '''
    Filter burst selections by their discussion string.

    Parameters
    ----------
    data : dict
        Selections to be combined. Must have key 'discussion'.
    filter : str
        Regular expression used to filter the data
    case_sensitive : bool
        Make the filter case sensitive
    '''
    # Make case-insensitive searches the default
    flags = re.IGNORECASE
    if case_sensitive:
        flags = 0

    return [seg
            for seg in data
            if re.search(filter, seg.discussion, flags)]


def plot_metric(ref_data, test_data, fig, labels, location,
                nbins=10):
    '''
    Visualize the overlap between segments.

    Parameters
    ----------
    ref_data : list of `BurstSegment`s
        Reference burst segments
    test_data : list of `BurstSegment`s
        Test burst segments. Determine which test segments
        overlap with the reference segments and by how much
    labels : tuple of str
        Labels for the reference and test segments
    location : tuple
        Location of the figure (row, col, nrows, ncols)
    nbins : int
        Number of histogram bins to create

    Returns:
    --------
    ax : `matplotlib.pyplot.Axes`
        Axes in which data is displayed
    ref_test_data : list of `BurstSegment`s
        Reference data that falls within the [start, stop] times
        of the test data.
    '''

    # Determine by how much the test data overlaps with
    # the reference data.
    ref_test = []
    ref_test_data = []
    ref_test = [selection_overlap(segment, test_data)
                for segment in ref_data]

    # Overlap statistics
    #   - Number of segments selected
    #   - Percentage of segments selected
    #   - Percent overlap from each segment
    ref_test_selected = sum(selection['n_selections'] > 0
                            for selection in ref_test)
    ref_test_pct_selected = ref_test_selected / len(ref_test) * 100.0
    ref_test_pct_overlap = [selection['pct_overlap'] for selection in ref_test]

    # Calculate the plot index from the (row,col) subplot location
    plot_idx = lambda rowcol, ncols: (rowcol[0] - 1) * ncols + rowcol[1]

    # Create a figure
    ax = fig.add_subplot(location[2], location[3],
                         plot_idx(location[0:2], location[3]))
    hh = ax.hist(ref_test_pct_overlap, bins=nbins, range=(0, 100))
    # ax.set_xlabel('% Overlap Between {0} and {1} Segments'.format(*labels))
    if location[0] == location[2]:
        ax.set_xlabel('% Overlap per Segment')
    if location[1] == 1:
        ax.set_ylabel('Occurrence')
    ax.text(0.5, 0.98, '{0:4.1f}% of {1:d}'
            .format(ref_test_pct_selected, len(ref_test)),
            verticalalignment='top', horizontalalignment='center',
            transform=ax.transAxes)
    ax.set_title('{0} Segments\nSelected by {1}'.format(*labels))

    return ax, ref_test


def metric(sroi=None, output_dir=None, figtype=None):
    do_sroi = False
    if sroi in (1, 2, 3):
        do_sroi = True

    if output_dir is None:
        output_dir = pathlib.Path('~/').expanduser()
    else:
        output_dir = pathlib.Path(output_dir).expanduser()

    starttime = dt.datetime(2019, 10, 17)

    # Find SROI
    # start_date, end_date = gls_get_sroi(starttime)
    start_date = dt.datetime(2019, 10, 19)
    # end_date = start_date + dt.timedelta(days=5)
    end_date = dt.datetime.now()

    abs_data = selections('abs', start_date, end_date,
                          latest=True, combine=True, metadata=do_sroi)

    gls_data = selections('mp-dl-unh', start_date, end_date,
                          latest=True, combine=True, metadata=do_sroi)

    sitl_data = selections('sitl+back', start_date, end_date,
                           latest=True, combine=True, metadata=do_sroi)

    # Filter by SROI
    if do_sroi:
        abs_data = [s for s in abs_data if s.sroi == sroi]
        sitl_data = [s for s in sitl_data if s.sroi == sroi]
        gls_data = [s for s in gls_data if s.sroi == sroi]

    sitl_mp_data = filter_segments(sitl_data, '(MP|Magnetopause)')

    # Create a figure
    nbins = 10
    nrows = 4
    ncols = 3
    fig = plt.figure(figsize=(8.5, 10))
    fig.subplots_adjust(hspace=0.55, wspace=0.3)

    # GLS-SITL Comparison
    ax, gls_sitl = plot_metric(gls_data, sitl_data, fig,
                               ('GLS', 'SITL'), (1, 1, nrows, ncols),
                               nbins=nbins)
    ax, sitl_gls = plot_metric(sitl_data, gls_data, fig,
                               ('SITL', 'GLS'), (2, 1, nrows, ncols),
                               nbins=nbins)
    ax, gls_sitl_mp = plot_metric(gls_data, sitl_mp_data, fig,
                                  ('GLS', 'SITL MP'), (3, 1, nrows, ncols),
                                  nbins=nbins)
    ax, sitl_mp_gls = plot_metric(sitl_mp_data, gls_data, fig,
                                  ('SITL MP', 'GLS'), (4, 1, nrows, ncols),
                                  nbins=nbins)

    # ABS-SITL Comparison
    ax, abs_sitl = plot_metric(abs_data, sitl_data, fig,
                               ('ABS', 'SITL'), (1, 2, nrows, ncols),
                               nbins=nbins)
    ax, sitl_abs = plot_metric(sitl_data, abs_data, fig,
                               ('SITL', 'ABS'), (2, 2, nrows, ncols),
                               nbins=nbins)
    ax, abs_sitl_mp = plot_metric(abs_data, sitl_mp_data, fig,
                                  ('ABS', 'SITL MP'), (3, 2, nrows, ncols),
                                  nbins=nbins)
    ax, sitl_mp_abs = plot_metric(sitl_mp_data, abs_data, fig,
                                  ('SITL MP', 'ABS'), (4, 2, nrows, ncols),
                                  nbins=nbins)

    # GLS-ABS Comparison
    abs_mp_data = [abs_data[idx]
                   for idx, s in enumerate(abs_sitl_mp)
                   if s['n_selections'] > 0]

    ax, gls_abs = plot_metric(gls_data, abs_data, fig,
                              ('GLS', 'ABS'), (1, 3, nrows, ncols),
                              nbins=nbins)
    ax, abs_gls = plot_metric(abs_data, gls_data, fig,
                              ('ABS', 'GLS'), (2, 3, nrows, ncols),
                              nbins=nbins)
    ax, gls_abs_mp = plot_metric(gls_data, abs_mp_data, fig,
                                 ('GLS', 'ABS MP'), (3, 3, nrows, ncols),
                                 nbins=nbins)
    ax, abs_mp_gls = plot_metric(abs_mp_data, gls_data, fig,
                                 ('ABS MP', 'GLS'), (4, 3, nrows, ncols),
                                 nbins=nbins)

    # Save the figure
    if figtype is not None:
        sroi_str = ''
        if do_sroi:
            sroi_str = '_sroi{0:d}'.format(sroi)
        filename = (output_dir
                    / '_'.join(('selections_metric' + sroi_str,
                                start_date.strftime('%Y%m%d%H%M%S'),
                                end_date.strftime('%Y%m%d%H%M%S')
                                )))
        filename = filename.with_suffix('.' + figtype)
        plt.savefig(filename.expanduser())

    plt.show()


def print_segments(data, full=False):
    '''
    Print details of the burst selections.

    Parameters
    ----------
    data : `BurstSegment` or list of `BurstSegment`
        Selections to be printed. Must have keys 'tstart', 'tstop',
        'fom', 'sourceid', and 'discussion'
    '''
    if full:
        source_len = max(len(s.sourceid) for s in data)
        source_len = max(source_len, 8)
        header_fmt = '{0:>19}   {1:>19}   {2:>19}   {3:>5}   ' \
                     '{4:>19}   {5:>' + str(source_len) + '}   {6}'
        data_fmt = '{0:>19}   {1:>19}   {2:>19}   {3:5.1f}   ' \
                   '{4:>19}, {5:>' + str(source_len) + '}   {6}'
        print(header_fmt.format('TSTART', 'TSTOP', 'CREATETIME',
                                'FOM', 'STATUS', 'SOURCEID', 'DISCUSSION'
                                )
              )
        for s in data:
            try:
                status = s.status
            except AttributeError:
                status = ''

            createtime = dt.datetime.strftime(s.createtime,
                                              '%Y-%m-%d %H:%M:%S')
            print(data_fmt.format(s.start_time, s.stop_time,
                                  createtime, s.fom, status, s.sourceid,
                                  s.discussion)
                  )
        return

    print('{0:>19}   {1:>19}   {2}   {3}'
          .format('TSTART', 'TSTOP', 'FOM', 'DISCUSSION')
          )

    if isinstance(data, list):
        for selection in data:
            print(selection)
    else:
        print(data)


def read_csv(filename, start_time=None, stop_time=None, header=True):
    '''
    Read a CSV file with burst segment selections.

    Parameters
    ----------
    filename : str
        The name of the file to which `data` is to be read
    start_time : str or `datetime.datetime`
        Filter results to contain segments selected on or
        after this time. Possible only if `header` is True
        and if a column is named `'start_time'`
    stop_time : str or `datetime.datetime`
        Filter results to contain segments selected on or
        before this time. Possible only if `header` is True
        and if a column is named `'stop_time'`
    header : bool
        If `True`, the csv file has a header indicating the
        names of each column. If `header` is `False`, the
        assumed column names are 'start_time', 'stop_time',
        'fom', 'sourceid', 'discussion', 'createtime'.

    Returns
    -------
    data : list of `BurstSegment`
        Burst segments read from the csv file
    '''
    # Convert time itnerval to datetimes if needed
    if isinstance(start_time, str):
        start_time = dt.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    if isinstance(stop_time, str):
        stop_time = dt.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')

    file = pathlib.Path(filename)
    data = []

    # Read the file
    with open(file.expanduser(), 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Take column names from file header
        if header:
            keys = next(csvreader)

        # Read the rows
        for row in csvreader:
            # Select the data within the time interval
            if start_time is not None:
                tstart = dt.datetime.strptime(row[0],
                                              '%Y-%m-%d %H:%M:%S')
                if tstart < start_time:
                    continue
            if stop_time is not None:
                tstop = dt.datetime.strptime(row[1],
                                             '%Y-%m-%d %H:%M:%S')
                if tstop > stop_time:
                    continue  # BREAK if sorted!!

            # Initialize segment with required fields then add
            # additional fields after
            data.append(BurstSegment(row[0], row[1], float(row[2]), row[4],
                                     sourceid=row[3], createtime=row[5]
                                     )
                        )

    return data


def remove_duplicate_segments(data):
    '''
    If SITL or GLS selections are submitted multiple times,
    there can be multiple copies of the same selection, or
    altered selections that overlap with what was selected
    previously. Find overlapping segments and select those
    from the most recent file, as indicated by the file name.

    Parameters
    ----------
    data : list of `BurstSegment`
        Selections from which to prude duplicates. Segments
        must be sorted by `tstart`.

    Returns
    -------
    results : list of `BurstSegments`
        Unique burst segments.
    '''
    results = data.copy()
    overwrite_times = set()
    for idx, segment in enumerate(data):
        iahead = idx + 1

        # Segments should already be sorted. Future segments
        # overlap with current segment if the future segement
        # start time is closer to the current segment start
        # time than is the current segment's end time.
        while ((iahead < len(data)) and
               ((data[iahead].tstart - segment.tstart) <
                (segment.tstop - segment.tstart))
        ):

            # Remove the segment with the earlier create time
            if segment.createtime < data[iahead].createtime:
                remove_segment = segment
                overwrite_time = segment.createtime
            else:
                remove_segment = data[iahead]
                overwrite_time = data[iahead].createtime

            # The segment may have already been removed if
            # there are more than one other segments that
            # overlap with it.
            try:
                results.remove(remove_segment)
                overwrite_times.add(overwrite_time)
            except ValueError:
                pass

            iahead += 1

    # Remove all segments with create times in the overwrite set
    # Note that the model results do not appear to be reproducible
    # so that every time the model is run, a different set of
    # selections are created. Using the filter below, data from
    # whole days can be removed if two processes create overlapping
    # segments at the edges of their time intervals.
    #     results = [segment
    #                for segment in results
    #                if segment.createtime not in overwrite_times
    #                ]

    print('# Segments Removed: {}'.format(len(data) - len(results)))
    return results


def remove_duplicate_segments_v1(data):
    idx = 0  # current index
    i = idx  # look-ahead index
    noverlap = 0
    result = []
    for idx in range(len(data)):
        if idx < i:
            continue
        print(idx)

        # Reference segment is the current segment
        ref_seg = data[idx]
        t0_ref = ref_seg.taistarttime
        t1_ref = ref_seg.taiendtime
        dt_ref = t1_ref - t0_ref

        if idx == len(data) - 2:
            import pdb
            pdb.set_trace()

        try:
            # Test segment are after the reference segment. The
            # test segment overlaps with the reference segment
            # if its start time is closer to the reference start
            # time than is the reference end time. If there is
            # overlap, keep the segment that was created more
            # recently.
            i = idx + 1
            while ((data[i].taistarttime - t0_ref) < dt_ref):
                noverlap += 1
                test_seg = data[i]
                if data[i].createtime > data[idx].createtime:
                    ref_seg = test_seg
                i += 1
        except IndexError:
            pass

        result.append(ref_seg)

    print('# Overlapping Segments: {}'.format(noverlap))
    return result


def selection_overlap(ref, tests):
    '''
    Gather overlap statistics.

    Parameters
    ----------
    ref : `selections.BurstSegment`
        The reference burst segment.
    tests : list of `selections.BurstSegment`
        The burst segements against which the reference segment is compared.

    Returns
    -------
    out : dict
        Data regarding how much the reference segment overlaps with the
        test segments.
    '''
    out = {'dt': ref.tstop - ref.tstart,
           'dt_next': dt.timedelta(days=7000),
           'n_selections': 0,
           't_overlap': dt.timedelta(seconds=0.0),
           't_overselect': dt.timedelta(seconds=0.0),
           'pct_overlap': 0.0,
           'pct_overselect': 0.0
           }

    # Find which selections overlap with the given entry and by how much
    tdelta = dt.timedelta(days=7000)
    for test in tests:

        if ((test.tstart <= ref.tstop) and
                (test.tstop >= ref.tstart)
        ):
            out['n_selections'] += 1
            out['t_overlap'] += (min(test.tstop, ref.tstop)
                                 - max(test.tstart, ref.tstart)
                                 )

        # Time to nearest interval
        out['dt_next'] = min(out['dt_next'], abs(test.tstart - ref.tstart))

    # Overlap and over-selection statistics
    if out['n_selections'] > 0:
        out['t_overselect'] = out['dt'] - out['t_overlap']
        out['pct_overlap'] = out['t_overlap'] / out['dt'] * 100.0
        out['pct_overselect'] = out['t_overselect'] / out['dt'] * 100.0
    else:
        out['t_overselect'] = out['dt']
        out['pct_overselect'] = 100.0

    return out


def selections(type, start, stop,
               sort=False, combine=False, latest=True, unique=False,
               metadata=False, filter=None, case_sensitive=False):
    '''
    Factory function for burst data selections.

    Parameters
    ----------
    type : str
        Type of burst data selections to retrieve. Options are 'abs',
        'abs-all', 'sitl', 'sitl+back', 'gls', 'mp-dl-unh'.
    start, stop : `datetime.datetime`
        Interval over which to retrieve data
    sort : bool
        Sort burst segments by time. Submissions to the back structure and
        multiple submissions or process executions in one SITL window can
        cause multiples of the same selectoion r out-of-order selections.
    combine : bool
        Combine adjacent selection into one selection. Due to downlink
        limitations, long time duration burst segments must be broken into
        smaller chunks.
    latest : bool
        For each SROI, keep only those times with the most recent time
        of creation. Cannot be used with `unique`.
    metadata : bool
        Retrieve the orbit number and time interval, SROI number and time
        interval, and SITL window time interval.
    unique : bool
        Return only unique segments. See `sort`. Also, a SITL may adjust
        the time interval of their selections, so different submissions will
        have duplicates selections but with different time stamps. This is
        accounted for.
    filter : str
        Filter the burst segments by applying the regular expression to
        the segment's discussions string.

    Returns
    -------
    results : list of `BurstSegment`
        Each burst segment.
    '''
    return _get_selections(type, start, stop,
                           sort=sort, combine=combine, unique=unique,
                           latest=latest, metadata=metadata,
                           filter=filter, case_sensitive=case_sensitive)


def sort_segments(data, createtime=False):
    '''
    Sort abs, sitl, or gls selections into ascending order.

    Parameters
    ----------
    data : list of `BurstSegement`
        Selections to be sorted. Must have keys 'start_time', 'end_time',
        'fom', 'discussion', 'tstart', and 'tstop'.
    createtime: bool
        Sort by time created instead of by selection start time.

    Returns
    -------
    results : list of `BurstSegement`
        Inputs sorted by time
    '''
    if createtime:
        return sorted(data, key=lambda x: x.createtime)
    return sorted(data, key=lambda x: x.tstart)


def write_csv(filename, data, append=False):
    '''
    Write a CSV file with burst data segment selections.

    Parameters
    ----------
    filename : str
        The name of the file to which `data` is to be written
    data : list of `BurstSegment`
        Burst segments to be written to the csv file
    append : bool
        If True, `data` will be appended to the end of the file.
    '''
    header = ('start_time', 'stop_time', 'fom', 'sourceid',
              'discussion', 'createtime')

    mode = 'w'
    if append:
        mode = 'a'

    file = pathlib.Path(filename)
    with open(file.expanduser(), mode, newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if not append:
            csvwriter.writerow(header)

        for segment in data:
            csvwriter.writerow([segment.start_time,
                                segment.stop_time,
                                segment.fom,
                                segment.sourceid,
                                segment.discussion,
                                segment.createtime
                                ]
                               )


if __name__ == 'main':
    from heliopy import config
    import pathlib

    # Inputs
    sc = sys.argv[0]
    start_date = sys.argv[1]
    if len(sys.argv) == 3:
        dir = sys.argv[2]
    else:
        dir = pathlib.Path(config['download_dir']) / 'figures' / 'mms'

    start_date = dt.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')

    # Plot the data
    fig = plot_context(sc, start_date, dir=dir)















#instead of importing util we will paste util in and define selections and mrmms_sdc_api as functions before it

# the following are the functions from selections that we will need.


# the following are the functions from mrmms_sdc_api  that we will need.
import glob
import os
import io
import re
import requests
import csv
import pymms as pymms
from tqdm import tqdm
import datetime as dt
import numpy as np
from cdflib import epochs
from urllib.parse import parse_qs
import urllib3
import warnings
from scipy.io import readsav
from getpass import getpass
from pathlib import Path


# data_root = pymms.config['data_root']
# dropbox_root = pymms.config['dropbox_root']
# mirror_root = pymms.config['mirror_root']
data_root = Path(""" /Users/Riley/PycharmProjects/mms_data/heliopy/heliopy/data/mms """)
#dropbox_root = ~/data/mms/dropbox
mirror_root = None

mms_username = None
mms_password = None


class MrMMS_SDC_API:
    """Interface with NASA's MMS SDC API

    Interface with the Science Data Center (SDC) API of the
    Magnetospheric Multiscale (MMS) mission.
    https://lasp.colorado.edu/mms/sdc/public/

    Params:
        sc (str,list):       Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')
        instr (str,list):    Instrument IDs
        mode (str,list):     Data rate mode ('slow', 'fast', 'srvy', 'brst')
        level (str,list):    Data quality level ('l1a', 'l1b', 'sitl', 'l2pre', 'l2', 'l3')
        data_type (str):     Type of data ('ancillary', 'hk', 'science')
        end_date (str):      End date of data interval, formatted as either %Y-%m-%d or
                             %Y-%m-%dT%H:%M:%S.
        files (str,list):    File names. If set, automatically sets `sc`, `instr`, `mode`,
                             `level`, `optdesc` and `version` to None.
        offline (bool):      Do not search for file information online.
        optdesc (str,list):  Optional file name descriptor
        site (str):          SDC site to use ('public', 'private'). Setting `level`
                             automatically sets `site`. If `level` is 'l2' or 'l3', then
                             `site`='public' otherwise `site`='private'.
        start_date (str):    Start date of data interval, formatted as either %Y-%m-%d or
                             %Y-%m-%dT%H:%M:%S.
        version (str,list):  File version numbers.
    """

    def __init__(self, sc=None, instr=None, mode=None, level=None,
                 data_type='science',
                 end_date=None,
                 files=None,
                 offline=False,
                 optdesc=None,
                 site='public',
                 start_date=None,
                 version=None):

        # Set attributes
        #   - Put site before level because level will auto-set site
        #   - Put files last because it will reset most fields
        self.site = site

        self.data_type = data_type
        self.end_date = end_date
        self.instr = instr
        self.level = level
        self.mode = mode
        self.offline = offline
        self.optdesc = optdesc
        self.sc = sc
        self.start_date = start_date
        self.version = version

        self.files = files

        self._data_root = data_root
        #self._dropbox_root = dropbox_root
        self._mirror_root = mirror_root
        self._sdc_home = 'https://lasp.colorado.edu/mms/sdc'
        self._info_type = 'download'

        # Create a persistent session
        self._session = requests.Session()
        if (mms_username is not None) and (mms_password is not None):
            self._session.auth = (mms_username, mms_password)

    def __str__(self):
        return self.url()

    # https://stackoverflow.com/questions/17576009/python-class-property-use-setter-but-evade-getter
    def __setattr__(self, name, value):
        """Control attribute values as they are set."""

        # TYPE OF INFO
        #   - Unset other complementary options
        #   - Ensure that at least one of (download | file_names |
        #     version_info | file_info) are true
        if name == 'data_type':
            if 'gls_selections' in value:
                if value[15:] not in ('mp-dl-unh',):
                    raise ValueError('Unknown GLS Selections type.')
            elif value not in ('ancillary', 'hk', 'science',
                               'abs_selections', 'sitl_selections',
                               'bdm_sitl_changes'):
                raise ValueError('Invalid value {} for attribute'
                                 ' "{}".'.format(value, name))

            # Unset attributes related to data_type = 'science'
            if 'selections' in value:
                self.sc = None
                self.instr = None
                self.mode = None
                self.level = None
                self.optdesc = None
                self.version = None

        elif name == 'files':
            if value is not None:
                # Keep track of site because setting
                # self.level = None will set self.site = 'public'
                site = self.site
                self.sc = None
                self.instr = None
                self.mode = None
                self.level = None
                self.optdesc = None
                self.version = None
                self.site = site

        elif name == 'level':
            # L2 and L3 are the only public data levels
            if value in [None, 'l2', 'l3']:
                self.site = 'public'
            else:
                self.site = 'private'

        elif name == 'site':
            # Team site is most commonly referred to as the "team",
            # or "private" site, but in the URL is referred to as the
            # "sitl" site. Accept any of these values.
            if value in ('private', 'team', 'sitl'):
                value = 'sitl'
            elif value == 'public':
                value = 'public'
            else:
                raise ValueError('Invalid value for attribute {}.'
                                 .format(name)
                                 )

        elif name in ('start_date', 'end_date'):
            # Convert string to datetime object
            if isinstance(value, str):
                try:
                    value = dt.datetime.strptime(value[0:19],
                                                 '%Y-%m-%dT%H:%M:%S'
                                                 )
                except ValueError:
                    try:
                        value = dt.datetime.strptime(value, '%Y-%m-%d')
                    except ValueError:
                        raise

        # Set the value
        super(MrMMS_SDC_API, self).__setattr__(name, value)

    def url(self, query=True):
        """
        Build a URL to query the SDC.

        Parameters
        ----------
        query : bool
            If True (default), add the query string to the url.

        Returns
        -------
        url : str
            URL used to retrieve information from the SDC.
        """
        sep = '/'
        url = sep.join((self._sdc_home, self.site, 'files', 'api', 'v1',
                        self._info_type, self.data_type))

        # Build query from parts of file names
        if query:
            query_string = '?'
            qdict = self.query()
            for key in qdict:
                query_string += key + '=' + qdict[key] + '&'

            # Combine URL with query string
            url += query_string

        return url

    def check_response(self, response):
        '''
        Check the status code for a requests response and perform
        and appropriate action (e.g. log-in, raise error, etc.)

        Parameters
        ----------
        response : `requests.response`
            Response from the SDC

        Returns
        -------
        r : `requests.response`
            Updated response
        '''

        # OK
        if response.status_code == 200:
            r = response

        # Authentication required
        elif response.status_code == 401:
            print('Log-in Required')

            maxAttempts = 4
            nAttempts = 1
            while nAttempts <= maxAttempts:
                # First time through will automatically use the
                # log-in information from the config file. If that
                # information is wrong/None, ask explicitly
                if nAttempts == 1:
                    self.login(mms_username, mms_password)
                else:
                    self.login()

                # Remake the request
                #   - Ideally, self._session.send(response.request)
                #   - However, the prepared request lacks the
                #     authentication data
                if response.request.method == 'POST':
                    query = parse_qs(response.request.body)
                    r = self._session.post(response.request.url, data=query)
                else:
                    r = self._session.get(response.request.url)

                # Another attempt
                if r.ok:
                    break
                else:
                    print('Incorrect username or password. %d tries '
                          'remaining.' % maxAttempts - nAttempts)
                    nAttempts += 1

            # Failed log-in
            if nAttempts > maxAttempts:
                raise ConnectionError('Failed log-in.')

        else:
            raise ConnectionError(response.reason)

        # Return the resulting request
        return r

    def download_files(self):
        '''
        Download files from the SDC. First, search the local file
        system to see if they have already been downloaded.

        Returns
        -------
        local_files : list
            Names of the local files.
        '''

        # Get available files
        local_files, remote_files = self.search()
        if self.offline:
            return local_files

        # Download remote files
        #   - file_info() does not want the remote path
        if len(remote_files) > 0:
            remote_files = [file.split('/')[-1] for file in remote_files]
            downloaded_files = self.download_from_sdc(remote_files)
            local_files.extend(downloaded_files)

        return local_files

    def download_from_sdc(self, file_names):
        '''
        Download multiple files from the SDC. To prevent downloading the
        same file multiple times and to properly filter by file start time
        see the download_files method.

        Parameters
        ----------
        file_names : str, list
            File names of the data files to be downloaded. See
            the file_names method.

        Returns
        -------
        local_files : list
            Names of the local files. Remote files downloaded
            only if they do not already exist locally
        '''

        # Make sure files is a list
        if isinstance(file_names, str):
            file_names = [file_names]

        # Get information on the files that were found
        #   - To do that, specify the specific files.
        #     This sets all other properties to None
        #   - Save the state of the object as it currently
        #     is so that it can be restored
        #   - Setting FILES will indirectly cause SITE='public'.
        #     Keep track of SITE.
        state = {}
        state['sc'] = self.sc
        state['instr'] = self.instr
        state['mode'] = self.mode
        state['level'] = self.level
        state['optdesc'] = self.optdesc
        state['version'] = self.version
        state['files'] = self.files

        # Get file name and size
        self.files = file_names
        file_info = self.file_info()

        # Build the URL sans query
        self._info_type = 'download'
        url = self.url(query=False)

        # Amount to download per iteration
        block_size = 1024 * 128
        local_file_names = []

        # Download each file individually
        for info in file_info['files']:
            # Create the destination directory
            file = self.name2path(info['file_name'])
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            # downloading: https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
            # progress bar: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
            try:
                r = self._session.get(url,
                                      params={'file': info['file_name']},
                                      stream=True)
                with tqdm(total=info['file_size'],
                          unit='B',
                          unit_scale=True,
                          unit_divisor=1024
                          ) as pbar:
                    with open(file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                pbar.update(block_size)
            except:
                if os.path.isfile(file):
                    os.remove(file)
                for key in state:
                    self.files = None
                    setattr(self, key, state[key])
                raise

            local_file_names.append(file)

        # Restore the entry state
        self.files = None
        for key in state:
            setattr(self, key, state[key])

        return local_file_names

    def download_from_sdc_v1(self, file_names):
        '''
        Download multiple files from the SDC. To prevent downloading the
        same file multiple times and to properly filter by file start time
        see the download_files method.

        This version of the program calls `self.file_info()` for each
        file name given, whereas `download_from_sdc_v1` calls it once
        for all files. In the event of many files, `self.get()` was
        altered to use `requests.post()` instead of `requests.get()`
        if the url was too long (i.e. too many files).

        Parameters
        ----------
        file_names : str, list
            File names of the data files to be downloaded. See
            the file_names method.

        Returns
        -------
        local_files : list
            Names of the local files. Remote files downloaded
            only if they do not already exist locally
        '''

        # Make sure files is a list
        if isinstance(file_names, str):
            file_names = [file_names]

        # Get information on the files that were found
        #   - To do that, specify the specific files.
        #     This sets all other properties to None
        #   - Save the state of the object as it currently
        #     is so that it can be restored
        #   - Setting FILES will indirectly cause SITE='public'.
        #     Keep track of SITE.
        site = self.site
        state = {}
        state['sc'] = self.sc
        state['instr'] = self.instr
        state['mode'] = self.mode
        state['level'] = self.level
        state['optdesc'] = self.optdesc
        state['version'] = self.version
        state['files'] = self.files

        # Build the URL sans query
        self.site = site
        self._info_type = 'download'
        url = self.url(query=False)

        # Amount to download per iteration
        block_size = 1024 * 128
        local_file_names = []

        # Download each file individually
        for file_name in file_names:
            self.files = file_name
            info = self.file_info()['files'][0]

            # Create the destination directory
            file = self.name2path(info['file_name'])
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            # downloading: https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
            # progress bar: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
            try:
                r = self._session.get(url,
                                      params={'file': info['file_name']},
                                      stream=True)
                with tqdm(total=info['file_size'],
                          unit='B',
                          unit_scale=True,
                          unit_divisor=1024
                          ) as pbar:
                    with open(file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                pbar.update(block_size)
            except:
                if os.path.isfile(file):
                    os.remove(file)
                for key in state:
                    self.files = None
                    setattr(self, key, state[key])
                raise

            local_file_names.append(file)

        # Restore the entry state
        self.files = None
        for key in state:
            setattr(self, key, state[key])

        return local_file_names

    def download(self):
        '''
        Download multiple files. First, search the local file system
        to see if any of the files have been downloaded previously.

        Returns
        -------
        local_files : list
            Names of the local files. Remote files downloaded
            only if they do not already exist locally
        '''
        warnings.warn('This method will be removed in the future. Use the get method.',
                      DeprecationWarning)

        self._info_type = 'download'
        # Build the URL sans query
        url = self.url(query=False)

        # Get available files
        local_files, remote_files = self.search()
        if self.offline:
            return local_files

        # Get information on the files that were found
        #   - To do that, specify the specific files. This sets all other
        #     properties to None
        #   - Save the state of the object as it currently is so that it can
        #     be restored
        #   - Setting FILES will indirectly cause SITE='public'. Keep track
        #     of SITE.
        site = self.site
        state = {}
        state['sc'] = self.sc
        state['instr'] = self.instr
        state['mode'] = self.mode
        state['level'] = self.level
        state['optdesc'] = self.optdesc
        state['version'] = self.version
        state['files'] = self.files
        self.files = [file.split('/')[-1] for file in remote_files]

        self.site = site
        file_info = self.file_info()

        # Amount to download per iteration
        block_size = 1024 * 128

        # Download each file individually
        for info in file_info['files']:
            # Create the destination directory
            file = self.name2path(info['file_name'])
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            # Downloading and progress bar:
            # https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
            # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
            try:
                r = self._session.post(url,
                                       data={'file': info['file_name']},
                                       stream=True)
                with tqdm(total=info['file_size'], unit='B', unit_scale=True,
                          unit_divisor=1024) as pbar:
                    with open(file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                pbar.update(block_size)
            except:
                if os.path.isfile(file):
                    os.remove(file)
                for key in state:
                    self.files = None
                    setattr(self, key, state[key])
                raise

            local_files.append(file)

        self.files = None
        for key in state:
            setattr(self, key, state[key])

        return local_files

    def file_info(self):
        '''
        Obtain file information from the SDC.

        Returns
        -------
        file_info : list
                    Information about each file.
        '''
        self._info_type = 'file_info'
        response = self.get()
        return response.json()

    def file_names(self):
        '''
        Obtain file names from the SDC. Note that the SDC accepts only
        start and end dates, not datetimes. Therefore the files returned
        by this function may lie outside the time interval of interest.
        For a more precise list of file names, use the search method or
        filter the files with filter_time.

        Returns
        -------
        file_names : list
            Names of the requested files.
        '''
        self._info_type = 'file_names'
        response = self.get()

        # If no files were found, the empty string is the response
        # Return [] instead of [''] so that len() is zero.
        if response.text == '':
            return []
        return response.text.split(',')

    def get(self):
        '''
        Retrieve information from the SDC.

        Returns
        -------
        r : `session.response`
            Response to the request posted to the SDC.
        '''
        # Build the URL sans query
        url = self.url(query=False)

        # Check on query
        #   - Use POST if the URL is too long
        r = self._session.get(url, params=self.query())
        if r.status_code == 414:
            r = self._session.post(url, data=self.query())

        # Check if everything is ok
        if not r.ok:
            r = self.check_response(r)

        # Return the response for the requested URL
        return r

    def local_file_names(self, mirror=False):
        '''
        Search for MMS files on the local system. Files must be
        located in an MMS-like directory structure.

        Parameters
        ----------
        mirror : bool
            If True, the local data directory is used as the
            root directory. Otherwise the mirror directory is
            used.

        Returns
        -------
        local_files : list
            Names of the local files
        '''

        # Search the mirror or local directory
        if mirror:
            data_root = self._mirror_root
        else:
            data_root = self._data_root

        # If no start or end date have been defined,
        #   - Start at beginning of mission
        #   - End at today's date
        start_date = self.start_date
        end_date = self.end_date

        # Create all dates between start_date and end_date
        deltat = dt.timedelta(days=1)
        dates = []
        while start_date <= end_date:
            dates.append(start_date.strftime('%Y%m%d'))
            start_date += deltat

        # Paths in which to look for files
        #   - Files of all versions and times within interval
        if 'selections' in self.data_type:
            paths = construct_path(data_type=self.data_type,
                                   root=data_root, files=True)
        else:
            paths = construct_path(self.sc, self.instr, self.mode, self.level,
                                   dates, optdesc=self.optdesc,
                                   root=data_root, files=True)

        # Search
        result = []
        pwd = os.getcwd()
        for path in paths:
            root = os.path.dirname(path)

            try:
                os.chdir(root)
            except FileNotFoundError:
                continue
            except:
                os.chdir(pwd)
                raise

            for file in glob.glob(os.path.basename(path)):
                result.append(os.path.join(root, file))

        os.chdir(pwd)

        return result

    def login(self, username=None, password=None):
        '''
        Log-In to the SDC

        Parameters
        ----------
        username (str):     Account username
        password (str):     Account password
        '''

        # Ask for inputs
        if username is None:
            username = input('username: ')

        if password is None:
            password = input('password: ')

        # Save credentials
        self._session.auth = (username, password)

    def name2path(self, filename):
        '''
        Convert remote file names to local file name.

        Directories of a remote file name are separated by the '/' character,
        as in a web address.

        Parameters
        ----------
        filename : str
            File name for which the local path is desired.

        Returns
        -------
        path : str
            Equivalent local file name. This is the location to
            which local files are downloaded.
        '''
        parts = filename.split('_')

        # burst data selection directories and file names are structured as
        #   - dirname:  sitl/[type]_selections/
        #   - basename: [type]_selections_[optdesc]_YYYY-MM-DD-hh-mm-ss.sav
        # To get year, index from end to skip optional descriptor
        if parts[1] == 'selections':
            path = os.path.join(self._data_root, 'sitl',
                                '_'.join(parts[0:2]),
                                filename)

        # Burst directories and file names are structured as:
        #   - dirname:  sc/instr/mode/level[/optdesc]/YYYY/MM/DD/
        #   - basename: sc_instr_mode_level[_optdesc]_YYYYMMDDhhmmss_vX.Y.Z.cdf
        # Index from end to catch the optional descriptor, if it exists
        elif parts[2] == 'brst':
            path = os.path.join(self._data_root, *parts[0:-2],
                                parts[-2][0:4], parts[-2][4:6],
                                parts[-2][6:8], filename)

        # Survey (slow,fast,srvy) directories and file names are structured as:
        #   - dirname:  sc/instr/mode/level[/optdesc]/YYYY/MM/
        #   - basename: sc_instr_mode_level[_optdesc]_YYYYMMDD_vX.Y.Z.cdf
        # Index from end to catch the optional descriptor, if it exists
        else:
            path = os.path.join(self._data_root, *parts[0:-2],
                                parts[-2][0:4], parts[-2][4:6], filename)

        return path

    def parse_file_names(self, filename):
        '''
        Parse an official MMS file name. MMS file names are formatted as
            sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf
        where
            sc:       spacecraft id
            instr:    instrument id
            mode:     data rate mode
            level:    data level
            optdesc:  optional filename descriptor
            tstart:   start time of file
            vX.Y.Z:   file version, with X, Y, and Z version numbers

        Parameters
        ----------
        filename : str
            An MMS file name

        Returns
        -------
        parts : tuple
            A tuples ordered as
                (sc, instr, mode, level, optdesc, tstart, version)
            If opdesc is not present in the file name, the output will
            contain the empty string ('').
        '''
        parts = os.path.basename(filename).split('_')

        # If the file does not have an optional descriptor,
        # put an empty string in its place.
        if len(parts) == 6:
            parts.insert(-2, '')

        # Remove the file extension ``.cdf''
        parts[-1] = parts[-1][0:-4]
        return tuple(parts)

    def post(self):
        '''
        Retrieve data from the SDC.

        Returns
        -------
        r : `session.response`
            Response to the request posted to the SDC.
        '''
        # Build the URL sans query
        url = self.url(query=False)

        # Check on query
        r = self._session.post(url, data=self.query())

        # Check if everything is ok
        if not r.ok:
            r = self.check_response(r)

        # Return the response for the requested URL
        return r

    def query(self):
        '''
        build a dictionary of key-value pairs that serve as the URL
        query string.

        Returns
        -------
        query : dict
            URL query
        '''

        # Adjust end date
        #   - The query takes '%Y-%m-%d' but the object allows
        #     '%Y-%m-%dT%H:%M:%S'
        #   - Further, the query is half-exclusive: [start, end)
        #   - If the dates are the same but the times are different, then
        #     files between self.start_date and self.end_date will not be
        #     found
        #   - In these circumstances, increase the end date by one day
        if self.end_date is not None:
            end_date = self.end_date.strftime('%Y-%m-%d')
            if self.start_date.date() == self.end_date.date() or \
                    self.end_date.time() != dt.time(0, 0, 0):
                end_date = (self.end_date + dt.timedelta(1)
                            ).strftime('%Y-%m-%d')

        query = {}
        if self.sc is not None:
            query['sc_id'] = self.sc if isinstance(self.sc, str) \
                else ','.join(self.sc)
        if self.instr is not None:
            query['instrument_id'] = self.instr \
                if isinstance(self.instr, str) \
                else ','.join(self.instr)
        if self.mode is not None:
            query['data_rate_mode'] = self.mode if isinstance(self.mode, str) \
                else ','.join(self.mode)
        if self.level is not None:
            query['data_level'] = self.level if isinstance(self.level, str) \
                else ','.join(self.level)
        if self.optdesc is not None:
            query['descriptor'] = self.optdesc \
                if isinstance(self.optdesc, str) \
                else ','.join(self.optdesc)
        if self.version is not None:
            query['version'] = self.version if isinstance(self.version, str) \
                else ','.join(self.version)
        if self.files is not None:
            query['files'] = self.files if isinstance(self.files, str) \
                else ','.join(self.files)
        if self.start_date is not None:
            query['start_date'] = self.start_date.strftime('%Y-%m-%d')
        if self.end_date is not None:
            query['end_date'] = end_date

        return query

    def remote2localnames(self, remote_names):
        '''
        Convert remote file names to local file names.

        Directories of a remote file name are separated by the '/' character,
        as in a web address.

        Parameters
        ----------
        remote_names : list
            Remote file names returned by FileNames.

        Returns
        -------
        local_names : list
            Equivalent local file name. This is the location to
            which local files are downloaded.
        '''
        # os.path.join() requires string arguments
        #   - str.split() return list.
        #   - Unpack with *: https://docs.python.org/2/tutorial/controlflow.html#unpacking-argument-lists
        local_names = list()
        for file in remote_names:
            local_names.append(os.path.join(self._data_root,
                                            *file.split('/')[2:]))

        if (len(remote_names) == 1) & (type(remote_names) == 'str'):
            local_names = local_names[0]

        return local_names

    def search(self):
        '''
        Search for files locally and at the SDC.

        Returns
        -------
        files : tuple
            Local and remote files within the interval, returned as
            (local, remote), where `local` and `remote` are lists.
        '''

        # Search locally if offline
        if self.offline:
            local_files = self.local_file_names()
            remote_files = []

        # Search remote first
        #   - SDC is definitive source of files
        #   - Returns most recent version
        else:
            remote_files = self.file_names()

            # Search for the equivalent local file names
            local_files = self.remote2localnames(remote_files)
            idx = [i for i, local in enumerate(local_files)
                   if os.path.isfile(local)
                   ]

            # Filter based on location
            local_files = [local_files[i] for i in idx]
            remote_files = [remote_files[i] for i in range(len(remote_files))
                            if i not in idx
                            ]

        # Filter based on time interval
        if len(local_files) > 0:
            local_files = filter_time(local_files,
                                      self.start_date,
                                      self.end_date
                                      )
        if len(remote_files) > 0:
            remote_files = filter_time(remote_files,
                                       self.start_date,
                                       self.end_date
                                       )

        return (local_files, remote_files)

    def version_info(self):
        '''
        Obtain version information from the SDC.

        Returns
        -------
        vinfo : dict
            Version information regarding the requested files
        '''
        self._info_type = 'version_info'
        response = self.post()
        return response.json()


def _datetime_to_list(datetime):
    return [datetime.year, datetime.month, datetime.day,
            datetime.hour, datetime.minute, datetime.second,
            datetime.microsecond // 1000, datetime.microsecond % 1000, 0
            ]


def datetime_to_tai(t_datetime):
    # Convert datetime to TAI
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    return tt2000_to_tai(datetime_to_tt2000(t_datetime))


def datetime_to_tt2000(t_datetime):
    # Convert datetime to TT2000
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    t_list = _datetime_to_list(t_datetime)
    return epochs.CDFepoch.compute_tt2000(t_list)


def tai_to_tt2000(t_tai):
    # Convert TAI to TT2000
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    return np.asarray(t_tai) * int(1e9) + t_1958


def tai_to_datetime(t_tai):
    # Convert TAI to datetime
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    return tt2000_to_datetime(tai_to_tt2000(t_tai))


def tt2000_to_tai(t_tt2000):
    # Convert TT2000 to TAI
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    return (t_tt2000 - t_1958) // int(1e9)


def tt2000_to_datetime(t_tt2000):
    # Convert datetime to TT2000
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    tepoch = epochs.CDFepoch()
    return tepoch.to_datetime(t_tt2000)


def _response_text_to_dict(text):
    # Read first line as dict keys. Cut text from TAI keys
    f = io.StringIO(text)
    reader = csv.reader(f, delimiter=',')

    # Create a dictionary from the header
    data = dict()
    for key in next(reader):

        # See sitl_selections()
        if key.startswith(('start_time', 'end_time')):
            match = re.search('((start|end)_time)_utc', key)
            key = match.group(1)

        # See burst_data_segments()
        elif key.startswith('TAI'):
            match = re.search('(TAI(START|END)TIME)', key)
            key = match.group(1)

        data[key.lower()] = []

    # Read remaining lines into columns
    keys = data.keys()
    for row in reader:
        for key, value in zip(keys, row):
            data[key].append(value)

    return data


def burst_data_segments(start_date, end_date,
                        team=False, username=None):
    """
    Get information about burst data segments. Burst segments that
    were selected in the back structure are available through this
    service, but not through `sitl_selections()`. Also, the time
    between contiguous segments is 10 seconds.

    Parameters
    ----------
    start_date : `datetime`
        Start date of time interval for which information is desired.
    end_date : `datetime`
        End date of time interval for which information is desired.
    team : bool=False
        If set, information will be taken from the team site
        (login required). Otherwise, it is take from the public site.

    Returns
    -------
    data : dict
        Dictionary of information about burst data segments
            datasegmentid
            taistarttime    - Start time of burst segment in
                              TAI sec since 1958-01-01
            taiendtime      - End time of burst segment in
                              TAI sec since 1958-01-01
            parametersetid
            fom             - Figure of merit given to the burst segment
            ispending
            inplaylist
            status          - Download status of the segment
            numevalcycles
            sourceid        - Username of SITL who selected the segment
            createtime      - Time the selections were submitted as datetime (?)
            finishtime      - Time the selections were downlinked as datetime (?)
            obs1numbufs
            obs2numbufs
            obs3numbufs
            obs4numbufs
            obs1allocbufs
            obs2allocbufs
            obs3allocbufs
            obs4allocbufs
            obs1remfiles
            obs2remfiles
            obs3remfiles
            obs4remfiles
            discussion      - Description given to segment by SITL
            dt              - Duration of burst segment in seconds
            tstart          - Start time of burst segment as datetime
            tstop           - End time of burst segment as datetime
    """

    # Convert times to TAI since 1958
    t0 = _datetime_to_list(start_date)
    t1 = _datetime_to_list(end_date)
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    t0 = int((epochs.CDFepoch.compute_tt2000(t0) - t_1958) // 1e9)
    t1 = int((epochs.CDFepoch.compute_tt2000(t1) - t_1958) // 1e9)

    # URL
    url_path = 'https://lasp.colorado.edu/mms/sdc/'
    url_path += 'sitl/latis/dap/' if team else 'public/service/latis/'
    url_path += 'mms_burst_data_segment.csv'

    # Query string
    query = {}
    query['TAISTARTTIME>'] = '{0:d}'.format(t0)
    query['TAIENDTIME<'] = '{0:d}'.format(t1)

    # Post the query
    #    cookies = None
    #    if team:
    #        cookies = sdc_login(username)

    # Get the log-in information
    sesh = requests.Session()
    r = sesh.get(url_path, params=query)
    if r.status_code != 200:
        raise ConnectionError('{}: {}'.format(r.status_code, r.reason))

    # Read first line as dict keys. Cut text from TAI keys
    data = _response_text_to_dict(r.text)

    # Convert to useful types
    types = ['int16', 'int64', 'int64', 'str', 'float32', 'int8',
             'int8', 'str', 'int32', 'str', 'datetime', 'datetime',
             'int32', 'int32', 'int32', 'int32', 'int32', 'int32',
             'int32', 'int32', 'int32', 'int32', 'int32', 'str']
    for key, type in zip(data, types):
        if type == 'str':
            pass
        elif type == 'datetime':
            data[key] = [dt.datetime.strptime(value,
                                              '%Y-%m-%d %H:%M:%S'
                                              )
                         if value != '' else value
                         for value in data[key]
                         ]
        else:
            data[key] = np.asarray(data[key], dtype=type)

    # Add useful tags
    #   - Number of seconds elapsed
    #   - TAISTARTIME as datetime
    #   - TAIENDTIME as datetime
    data['dt'] = data['taiendtime'] - data['taistarttime']

    # Convert TAISTART/ENDTIME to datetimes
    #    NOTE! If data['TAISTARTTIME'] is a scalar, this will not work
    #          unless everything after "in" is turned into a list
    data['tstart'] = [dt.datetime(
        *value[0:6], value[6] * 1000 + value[7]
    )
        for value in
        epochs.CDFepoch.breakdown_tt2000(
            data['taistarttime'] * int(1e9) + t_1958
        )
    ]
    data['tstop'] = [dt.datetime(
        *value[0:6], value[6] * 1000 + value[7]
    )
        for value in
        epochs.CDFepoch.breakdown_tt2000(
            data['taiendtime'] * int(1e9) + t_1958
        )
    ]
    data['start_time'] = [tstart.strftime('%Y-%m-%d %H:%M:%S')
                          for tstart in data['tstart']]
    data['stop_time'] = [tend.strftime('%Y-%m-%d %H:%M:%S')
                         for tend in data['tstop']]

    return data


def burst_selections(selection_type, start, stop):
    '''
    A factory function for retrieving burst selection data.

    Parameters
    ----------
    type : str
        The type of data to retrieve. Options include:
        Type       Source                     Description
        =========  =========================  =======================================
        abs        download_selections_files  ABS selections
        sitl       download_selections_files  SITL selections
        sitl+back  burst_data_segments        SITL and backstructure selections
        gls        download_selections_files  ground loop selections from 'mp-dl-unh'
        mp-dl-unh  download_selections_files  ground loop selections from 'mp-dl-unh'
        =========  ========================   =======================================
    start, stop : `datetime.datetime`
        Time interval for which data is to be retrieved

    Returns
    -------
    data : struct
        The requested data
    '''
    if isinstance(start, (int, np.integer)):
        orbit = mission_events('orbit', start, start)
        start = min(orbit['tstart'])
    if isinstance(stop, (int, np.integer)):
        orbit = mission_events('orbit', stop, stop)
        stop = max(orbit['tend'])

    data_retriever = _get_selection_retriever(selection_type)
    return data_retriever(start, stop)


def _get_selection_retriever(selection_type):
    '''
    Creator function for mission events data.

    Parameters
    ----------
    selections_type : str
        Type of data desired

    Returns
    -------
    func : function
        Function to generate the data
    '''
    if selection_type == 'abs':
        return _get_abs_data
    elif selection_type == 'sitl':
        return _get_sitl_data
    elif selection_type == 'sitl+back':
        return burst_data_segments
    elif selection_type in ('gls', 'mp-dl-unh'):
        return _get_gls_data
    else:
        raise ValueError('Burst selection type {} not recognized'
                         .format(selection_type))


def _get_abs_data(start, stop):
    '''
    Download and read Automated Burst Selections sav files.
    '''
    abs_files = download_selections_files('abs_selections',
                                          start_date=start, end_date=stop)
    return _read_fom_structures(abs_files)


def _get_sitl_data(start, stop):
    '''
    Download and read SITL selections sav files.
    '''
    sitl_files = download_selections_files('sitl_selections',
                                           start_date=start, end_date=stop)
    return _read_fom_structures(sitl_files)


def _get_gls_data(start, stop):
    '''
    Download and read Ground Loop Selections csv files.
    '''
    gls_files = download_selections_files('gls_selections',
                                          gls_type='mp-dl-unh',
                                          start_date=start, end_date=stop)

    # Prepare to loop over files
    if isinstance(gls_files, str):
        gls_files = [gls_files]

    # Statistics of bad selections
    fskip = 0  # number of files skipped
    nskip = 0  # number of selections skipped
    nexpand = 0  # number of selections expanded
    result = dict()

    # Read multiple files
    for file in gls_files:
        data = read_gls_csv(file)

        # Accumulative sum of errors
        fskip += data['errors']['fskip']
        nskip += data['errors']['nskip']
        nexpand += data['errors']['nexpand']
        if data['errors']['fskip']:
            continue
        del data['errors']

        # Extend results from all files. Keep track of the file
        # names since entries can change. The most recent file
        # contains the correct selections information.
        if len(result) == 0:
            result = data
            result['file'] = [file] * len(result['fom'])
        else:
            result['file'].extend([file] * len(result['fom']))
            for key, value in data.items():
                result[key].extend(value)

    # Display bad data
    if (fskip > 0) | (nskip > 0) | (nexpand > 0):
        print('GLS Selection Adjustments:')
        print('  # files skipped:    {}'.format(fskip))
        print('  # entries skipped:  {}'.format(nskip))
        print('  # entries expanded: {}'.format(nexpand))

    return result


def _read_fom_structures(files):
    '''
    Read multiple IDL sav files containing ABS or SITL selections.
    '''
    # Read data from all files
    result = dict()
    for file in files:
        data = read_eva_fom_structure(file)
        if data['valid'] == 0:
            print('Skipping invalid file {0}'.format(file))
            continue

        # Turn scalars into lists so they can be accumulated
        # across multiple files.
        #
        # Keep track of file name because the same selections
        # (or updated versions of the same selections) can be
        # stored in multiple files, if they were submitted to
        # the SDC multiple times.
        if len(result) == 0:
            result = {key:
                          (value
                           if isinstance(value, list)
                           else [value]
                           )
                      for key, value in data.items()
                      }
            result['file'] = [file] * len(data['fom'])

        # Append or extend data from subsequent files
        else:
            result['file'].extend([file] * len(data['fom']))
            for key, value in data.items():
                if isinstance(value, list):
                    result[key].extend(value)
                else:
                    result[key].append(value)

    return result


def construct_file_names(*args, data_type='science', **kwargs):
    '''
    Construct a file name compliant with MMS file name format guidelines.

    MMS file names follow the convention
        sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf

    Parameters
    ----------
        *args : dict
            Arguments to be passed along.
        data_type : str
            Type of file names to construct. Options are:
            science or *_selections. If science, inputs are
            passed to construct_science_file_names. If
            *_selections, inputs are passed to
            construct_selections_file_names.
        **kwargs : dict
            Keywords to be passed along.

    Returns
    -------
        fnames : list
            File names constructed from inputs.
    '''

    if data_type == 'science':
        fnames = construct_science_file_names(*args, **kwargs)
    elif 'selections' in data_type:
        fnames = construct_selections_file_names(data_type, **kwargs)

    return fnames


def construct_selections_file_names(data_type, tstart='*', gls_type=None):
    '''
    Construct a SITL selections file name compliant with
    MMS file name format guidelines.

    MMS SITL selection file names follow the convention
        data_type_[gls_type]_tstart.sav

    Parameters
    ----------
        data_type : str, list, tuple
            Type of selections. Options are abs_selections
            sitl_selections, or gls_selections.
        tstart : str, list
            Start time of data file. The format is
            YYYY-MM-DD-hh-mm-ss. If not given, the default is "*".
        gls_type : str, list
            Type of ground-loop selections. Possible values are:
            mp-dl-unh.

    Returns
    -------
        fnames : list
            File names constructed from inputs.
    '''

    # Convert inputs to iterable lists
    if isinstance(data_type, str):
        data_type = [data_type]
    if isinstance(gls_type, str):
        gls_type = [gls_type]
    if isinstance(tstart, str):
        tstart = [tstart]

    # Accept tuples, as those returned by Construct_Filename
    if isinstance(data_type, tuple):
        data_type = [file[0] for file in data_type]
        tstart = [file[-1] for file in data_type]

        if len(data_type > 2):
            gls_type = [file[1] for file in data_type]
        else:
            gls_type = None

    # Create the file names
    if gls_type is None:
        fnames = ['_'.join((d, g, t + '.sav'))
                  for d in data_type
                  for t in tstart
                  ]

    else:
        fnames = ['_'.join((d, g, t + '.sav'))
                  for d in data_type
                  for g in gls_type
                  for t in tstart
                  ]

    return fnames


def construct_science_file_names(sc, instr=None, mode=None, level=None,
                                 tstart='*', version='*', optdesc=None):
    '''
    Construct a science file name compliant with MMS
    file name format guidelines.

    MMS science file names follow the convention
        sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf

    Parameters
    ----------
        sc : str, list, tuple
            Spacecraft ID(s)
        instr : str, list
            Instrument ID(s)
        mode : str, list
            Data rate mode(s). Options include slow, fast, srvy, brst
        level : str, list
            Data level(s). Options include l1a, l1b, l2pre, l2, l3
        tstart : str, list
            Start time of data file. In general, the format is
            YYYYMMDDhhmmss for "brst" mode and YYYYMMDD for "srvy"
            mode (though there are exceptions). If not given, the
            default is "*".
        version : str, list
            File version, formatted as "X.Y.Z", where X, Y, and Z
            are integer version numbers.
        optdesc : str, list
            Optional file name descriptor. If multiple parts,
            they should be separated by hyphens ("-"), not under-
            scores ("_").

    Returns
    -------
        fnames : str, list
            File names constructed from inputs.
    '''

    # Convert all to lists
    if isinstance(sc, str):
        sc = [sc]
    if isinstance(instr, str):
        instr = [instr]
    if isinstance(mode, str):
        mode = [mode]
    if isinstance(level, str):
        level = [level]
    if isinstance(tstart, str):
        tstart = [tstart]
    if isinstance(version, str):
        version = [version]
    if optdesc is not None and isinstance(optdesc, str):
        optdesc = [optdesc]

    # Accept tuples, as those returned by Construct_Filename
    if type(sc) == 'tuple':
        sc_ids = [file[0] for file in sc]
        instr = [file[1] for file in sc]
        mode = [file[2] for file in sc]
        level = [file[3] for file in sc]
        tstart = [file[-2] for file in sc]
        version = [file[-1] for file in sc]

        if len(sc) > 6:
            optdesc = [file[4] for file in sc]
        else:
            optdesc = None
    else:
        sc_ids = sc

    if optdesc is None:
        fnames = ['_'.join((s, i, m, l, t, 'v' + v + '.cdf'))
                  for s in sc_ids
                  for i in instr
                  for m in mode
                  for l in level
                  for t in tstart
                  for v in version
                  ]
    else:
        fnames = ['_'.join((s, i, m, l, o, t, 'v' + v + '.cdf'))
                  for s in sc_ids
                  for i in instr
                  for m in mode
                  for l in level
                  for o in optdesc
                  for t in tstart
                  for v in version
                  ]
    return fnames


def construct_path(*args, data_type='science', **kwargs):
    '''
    Construct a directory structure compliant with MMS path guidelines.

    MMS paths follow the convention
        selections: sitl/type_selections_[gls_type_]
        brst: sc/instr/mode/level[/optdesc]/<year>/<month>/<day>
        srvy: sc/instr/mode/level[/optdesc]/<year>/<month>

    Parameters
    ----------
        *args : dict
            Arguments to be passed along.
        data_type : str
            Type of file names to construct. Options are:
            science or *_selections. If science, inputs are
            passed to construct_science_file_names. If
            *_selections, inputs are passed to
            construct_selections_file_names.
        **kwargs : dict
            Keywords to be passed along.

    Returns
    -------
    paths : list
        Paths constructed from inputs.
    '''

    if data_type == 'science':
        paths = construct_science_path(*args, **kwargs)
    elif 'selections' in data_type:
        paths = construct_selections_path(data_type, **kwargs)
    else:
        raise ValueError('Invalid value for keyword data_type')

    return paths


def construct_selections_path(data_type, tstart='*', gls_type=None,
                              root='', files=False):
    '''
    Construct a directory structure compliant with MMS path
    guidelines for SITL selections.

    MMS SITL selections paths follow the convention
        sitl/[data_type]_selections[_gls_type]/

    Parameters
    ----------
        data_type : str, list, tuple
            Type of selections. Options are abs_selections
            sitl_selections, or gls_selections.
        tstart : str, list
            Start time of data file. The format is
            YYYY-MM-DD-hh-mm-ss. If not given, the default is "*".
        gls_type : str, list
            Type of ground-loop selections. Possible values are:
            mp-dl-unh.
        root : str
            Root of the SDC-like directory structure.
        files : bool
            If True, file names are associated with each path.

    Returns
    -------
    paths : list
        Paths constructed from inputs.
    '''

    # Convert inputs to iterable lists
    if isinstance(data_type, str):
        data_type = [data_type]
    if isinstance(gls_type, str):
        gls_type = [gls_type]
    if isinstance(tstart, str):
        tstart = [tstart]

    # Accept tuples, as those returned by Construct_Filename
    if isinstance(data_type, tuple):
        data_type = [file[0] for file in data_type]
        tstart = [file[-1] for file in data_type]

        if len(data_type > 2):
            gls_type = [file[1] for file in data_type]
        else:
            gls_type = None

    # Paths + Files
    if files:
        if gls_type is None:
            paths = [os.path.join(root, 'sitl', d, '_'.join((d, t + '.sav')))
                     for d in data_type
                     for t in tstart
                     ]
        else:
            paths = [os.path.join(root, 'sitl', d, '_'.join((d, g, t + '.sav')))
                     for d in data_type
                     for g in gls_type
                     for t in tstart
                     ]

    # Paths
    else:
        if gls_type is None:
            paths = [os.path.join(root, 'sitl', d)
                     for d in data_type
                     ]
        else:
            paths = [os.path.join(root, 'sitl', d)
                     for d in data_type
                     ]

    return paths


def construct_science_path(sc, instr=None, mode=None, level=None, tstart='*',
                           optdesc=None, root='', files=False):
    '''
    Construct a directory structure compliant with
    MMS path guidelines for science files.

    MMS science paths follow the convention
        brst: sc/instr/mode/level[/optdesc]/<year>/<month>/<day>
        srvy: sc/instr/mode/level[/optdesc]/<year>/<month>

    Parameters
    ----------
        sc : str, list, tuple
            Spacecraft ID(s)
        instr : str, list
            Instrument ID(s)
        mode : str, list
            Data rate mode(s). Options include slow, fast, srvy, brst
        level : str, list
            Data level(s). Options include l1a, l1b, l2pre, l2, l3
        tstart : str, list
            Start time of data file, formatted as a date: '%Y%m%d'.
            If not given, all dates from 20150901 to today's date are
            used.
        optdesc : str, list
            Optional file name descriptor. If multiple parts,
            they should be separated by hyphens ("-"), not under-
            scores ("_").
        root : str
            Root directory at which the directory structure begins.
        files : bool
            If True, file names will be generated and appended to the
            paths. The file tstart will be "YYYYMMDD*" (i.e. the date
            with an asterisk) and the version number will be "*".

    Returns
    -------
    fnames : str, list
        File names constructed from inputs.
    '''

    # Convert all to lists
    if isinstance(sc, str):
        sc = [sc]
    if isinstance(instr, str):
        instr = [instr]
    if isinstance(mode, str):
        mode = [mode]
    if isinstance(level, str):
        level = [level]
    if isinstance(tstart, str):
        tstart = [tstart]
    if optdesc is not None and isinstance(optdesc, str):
        optdesc = [optdesc]

    # Accept tuples, as those returned by construct_filename
    if type(sc) == 'tuple':
        sc_ids = [file[0] for file in sc]
        instr = [file[1] for file in sc]
        mode = [file[2] for file in sc]
        level = [file[3] for file in sc]
        tstart = [file[-2] for file in sc]

        if len(sc) > 6:
            optdesc = [file[4] for file in sc]
        else:
            optdesc = None
    else:
        sc_ids = sc

    # Paths + Files
    if files:
        if optdesc is None:
            paths = [os.path.join(root, s, i, m, l, t[0:4], t[4:6], t[6:8],
                                  '_'.join((s, i, m, l, t + '*', 'v*.cdf'))
                                  )
                     if m == 'brst'
                     else
                     os.path.join(root, s, i, m, l, t[0:4], t[4:6],
                                  '_'.join((s, i, m, l, t + '*', 'v*.cdf'))
                                  )
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for t in tstart
                     ]
        else:
            paths = [os.path.join(root, s, i, m, l, o, t[0:4], t[4:6], t[6:8],
                                  '_'.join((s, i, m, l, o, t + '*', 'v*.cdf'))
                                  )
                     if m == 'brst'
                     else
                     os.path.join(root, s, i, m, l, o, t[0:4], t[4:6],
                                  '_'.join((s, i, m, l, o, t + '*', 'v*.cdf'))
                                  )
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for o in optdesc
                     for t in tstart
                     ]

    # Paths
    else:
        if optdesc is None:
            paths = [os.path.join(root, s, i, m, l, t[0:4], t[4:6], t[6:8])
                     if m == 'brst' else
                     os.path.join(root, s, i, m, l, t[0:4], t[4:6])
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for t in tstart
                     ]
        else:
            paths = [os.path.join(root, s, i, m, l, o, t[0:4], t[4:6], t[6:8])
                     if m == 'brst' else
                     os.path.join(root, s, i, m, l, o, t[0:4], t[4:6])
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for o in optdesc
                     for t in tstart
                     ]

    return paths


def download_selections_files(data_type='abs_selections',
                              start_date=None, end_date=None,
                              gls_type=None):
    """
    Download SITL selections from the SDC.

    Parameters
    ----------
    data_type : str
        Type of SITL selections to download. Options are
            'abs_selections', 'sitl_selections', 'gls_selections'
    gls_type : str
        Type of gls_selections. Options are
            'mp-dl-unh'
    start_date : `dt.datetime` or str
        Start date of data interval
    end_date : `dt.datetime` or str
        End date of data interval

    Returns
    -------
    local_files : list
        Names of the selection files that were downloaded. Files
        can be read using mms.read_eva_fom_structure()
    """

    if gls_type is not None:
        data_type = '_'.join((data_type, gls_type))

    # Setup the API
    api = MrMMS_SDC_API()
    api.data_type = data_type
    api.start_date = start_date
    api.end_date = end_date

    # Download the files
    local_files = api.download_files()
    return local_files


def file_start_time(file_name):
    '''
    Extract the start time from a file name.

    Parameters
    ----------
        file_name : str
            File name from which the start time is extracted.

    Returns
    -------
        fstart : `datetime.datetime`
            Start time of the file, extracted from the file name
    '''

    try:
        # Selections: YYYY-MM-DD-hh-mm-ss
        fstart = re.search('[0-9]{4}(-[0-9]{2}){5}', file_name).group(0)
        fstart = dt.datetime.strptime(fstart, '%Y-%m-%d-%H-%M-%S')
    except AttributeError:
        try:
            # Brst: YYYYMMDDhhmmss
            fstart = re.search('20[0-9]{2}'  # Year
                               '(0[0-9]|1[0-2])'  # Month
                               '([0-2][0-9]|3[0-1])'  # Day
                               '([0-1][0-9]|2[0-4])'  # Hour
                               '[0-5][0-9]'  # Minute
                               '([0-5][0-9]|60)',  # Second
                               file_name).group(0)
            fstart = dt.datetime.strptime(fstart, '%Y%m%d%H%M%S')
        except AttributeError:
            try:
                # Srvy: YYYYMMDD
                fstart = re.search('20[0-9]{2}'  # Year
                                   '(0[0-9]|1[0-2])'  # Month
                                   '([0-2][0-9]|3[0-1])',  # Day
                                   file_name).group(0)
                fstart = dt.datetime.strptime(fstart, '%Y%m%d')
            except AttributeError:
                raise AttributeError('File start time not identified in: \n'
                                     '  "{}"'.format(file_name))

    return fstart


def filename2path(fname, root=''):
    """
    Convert an MMS file name to an MMS path.

    MMS paths take the form

        sc/instr/mode/level[/optdesc]/YYYY/MM[/DD/]

    where the optional descriptor [/optdesc] is included if it is also in the
    file name and day directory [/DD] is included if mode='brst'.

    Parameters
    ----------
    fname : str
        File name to be turned into a path.
    root : str
        Absolute directory

    Returns
    -------
    path : list
        Path to the data file.
    """

    parts = parse_file_name(fname)

    # data_type = '*_selections'
    if 'selections' in parts[0]:
        path = os.path.join(root, parts[0])

    # data_type = 'science'
    else:
        # Create the directory structure
        #   sc/instr/mode/level[/optdesc]/YYYY/MM/
        path = os.path.join(root, *parts[0:5], parts[5][0:4], parts[5][4:6])

        # Burst files require the DAY directory
        #   sc/instr/mode/level[/optdesc]/YYYY/MM/DD/
        if parts[2] == 'brst':
            path = os.path.join(path, parts[5][6:8])

    path = os.path.join(path, fname)

    return path


def filter_time(fnames, start_date, end_date):
    """
    Filter files by their start times.

    Parameters
    ----------
    fnames : str, list
        File names to be filtered.
    start_date : str
        Start date of time interval, formatted as '%Y-%m-%dT%H:%M:%S'
    end_date : str
        End date of time interval, formatted as '%Y-%m-%dT%H:%M:%S'

    Returns
    -------
    paths : list
        Path to the data file.
    """

    # Make sure file names are iterable. Allocate output array
    files = fnames
    if isinstance(files, str):
        files = [files]

    # If dates are strings, convert them to datetimes
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')

    # Parse the time out of the file name
    fstart = [file_start_time(file) for file in files]

    # Sort the files by start time
    isort = sorted(range(len(fstart)), key=lambda k: fstart[k])
    fstart = [fstart[i] for i in isort]
    files = [files[i] for i in isort]

    # End time
    #   - Any files that start on or before END_DATE can be kept
    idx = [i for i, t in enumerate(fstart) if t <= end_date]
    if len(idx) > 0:
        fstart = [fstart[i] for i in idx]
        files = [files[i] for i in idx]
    else:
        fstart = []
        files = []

    # Start time
    #   - Any file with TSTART <= START_DATE can potentially have data
    #     in our time interval of interest.
    #   - Assume the start time of one file marks the end time of the
    #     previous file.
    #   - With this, we look for the file that begins just prior to START_DATE
    #     and throw away any files that start before it.
    idx = [i for i, t in enumerate(fstart) if t >= start_date]
    if (len(idx) == 0) and \
            (len(fstart) > 0) and \
            (fstart[-1].date() == start_date.date()):
        idx = [len(fstart) - 1]

    elif (len(idx) != 0) and \
            ((idx[0] != 0) and (fstart[idx[0]] != start_date)):
        idx.insert(0, idx[0] - 1)

    if len(idx) > 0:
        fstart = [fstart[i] for i in idx]
        files = [files[i] for i in idx]
    else:
        fstart = []
        files = []

    return files


def filter_version(files, latest=None, version=None, min_version=None):
    '''
    Filter file names according to their version numbers.

    Parameters
    ----------
    files : str, list
        File names to be turned into paths.
    latest : bool
        If True, the latest version of each file type is
        returned. if `version` and `min_version` are not
        set, this is the default.
    version : str
        Only files with this version are returned.
    min_version : str
        All files with version greater or equal to this
        are returned.

    Returns
    -------
    filtered_files : list
        The files remaining after applying filter conditions.
    '''

    if version is None and min is None:
        latest = True
    if ((version is None) + (min_version is None) + (latest is None)) > 1:
        ValueError('latest, version, and min are mutually exclusive.')

    # Output list
    filtered_files = []

    # Extract the version
    parts = [parse_file_name(file) for file in files]
    versions = [part[-1] for part in parts]

    # The latest version of each file type
    if latest:
        # Parse file names and identify unique file types
        #   - File types include all parts of file name except version number
        bases = ['_'.join(part[0:-2]) for part in parts]
        uniq_bases = list(set(bases))

        # Filter according to unique file type
        for idx, uniq_base in enumerate(uniq_bases):
            test_idx = [i
                        for i, test_base in enumerate(bases)
                        if test_base == uniq_base]
            file_ref = files[idx]
            vXYZ_ref = [int(v) for v in versions[idx].split('.')]

            filtered_files.append(file_ref)
            for i in test_idx:
                vXYZ = [int(v) for v in versions[i].split('.')]
                if ((vXYZ[0] > vXYZ_ref[0]) or
                        (vXYZ[0] == vXYZ_ref[0] and
                         vXYZ[1] > vXYZ_ref[1]) or
                        (vXYZ[0] == vXYZ_ref[0] and
                         vXYZ[1] == vXYZ_ref[1] and
                         vXYZ[2] > vXYZ_ref[2])):
                    filtered_files[-1] = files[i]

    # All files with version number greater or equal to MIN_VERSION
    elif min_version is not None:
        vXYZ_min = [int(v) for v in min_version.split('.')]
        for idx, v in enumerate(versions):
            vXYZ = [int(vstr) for vstr in v.split('.')]
            if ((vXYZ[0] > vXYZ_min[0]) or
                    ((vXYZ[0] == vXYZ_min[0]) and
                     (vXYZ[1] > vXYZ_min[1])) or
                    ((vXYZ[0] == vXYZ_min[0]) and
                     (vXYZ[1] == vXYZ_min[1]) and
                     (vXYZ[2] >= vXYZ_min[2]))):
                filtered_files.append(files[idx])

    # All files with a particular version number
    elif version is not None:
        vXYZ_ref = [int(v) for v in version.split('.')]
        for idx, v in enumerate(versions):
            vXYZ = [int(vstr) for vstr in v.split('.')]
            if (vXYZ[0] == vXYZ_ref[0] and
                    vXYZ[1] == vXYZ_ref[1] and
                    vXYZ[2] == vXYZ_ref[2]):
                filtered_files.append(files[idx])

    return filtered_files


def mission_events(event_type, start, stop, sc=None):
    """
    Download MMS mission events. See the filters on the webpage
    for more ideas.
        https://lasp.colorado.edu/mms/sdc/public/about/events/#/

    Parameters
    ----------
    event_type : str
        Type of event. Options are 'apogee', 'dsn_contact', 'orbit',
        'perigee', 'science_roi', 'shadow', 'sitl_window', 'sroi'.
    start, stop : `datetime.datetime`, int
        Start and end of the data interval, specified as a time or
        orbit range.
    sc : str
        Spacecraft ID (mms, mms1, mms2, mms3, mms4) for which event
        information is to be returned.

    Returns
    -------
    data : dict
        Information about each event.
            start_time     - Start time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            end_time       - End time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            event_type     - Type of event
            sc_id          - Spacecraft to which the event applies
            source         - Source of event
            description    - Description of event
            discussion
            start_orbit    - Orbit on which the event started
            end_orbit      - Orbit on which the event ended
            tag
            id
            tstart         - Start time of event as datetime
            tend           - end time of event as datetime
    """
    event_func = _get_mission_events(event_type)
    return event_func(start, stop, sc)


def _get_mission_events(event_type):
    if event_type == 'apogee':
        return _get_apogee
    elif event_type == 'dsn_contact':
        return _get_dsn_contact
    elif event_type == 'orbit':
        return _get_orbit
    elif event_type == 'perigee':
        return _get_perigee
    elif event_type == 'science_roi':
        return _get_science_roi
    elif event_type == 'shadow':
        return _get_shadow
    elif event_type == 'sitl_window':
        return _get_sitl_window
    elif event_type == 'sroi':
        return _get_sroi


def _get_apogee(start, stop, sc):
    '''
    Apogee information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='apogee')


def _get_dsn_contact(start, stop, sc):
    '''
    Science region of interest information between `start` and `stop`
    and associated with spacecraft `sc`. Defines the limits of when
    fast survey and burst data can be available each orbit.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='dsn_contact')


def _get_orbit(start, stop, sc):
    '''
    Orbital information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='orbit')


def _get_perigee(start, stop, sc):
    '''
    Perigee information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='perigee')


def _get_science_roi(start, stop, sc):
    '''
    Science region of interest information between `start` and `stop`
    and associated with spacecraft `sc`. Defines the limits of when
    fast survey and burst data can be available each orbit.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='BDM', event_type='science_roi')


def _get_shadow(start, stop, sc):
    '''
    Earth shadow information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='POC', event_type='shadow')


def _get_sroi(start, stop, sc):
    '''
    Sub-region of interest information between `start` and `stop`
    and associated with spacecraft `sc`. There can be several
    SROIs per science_roi.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='POC', event_type='SROI')


def _get_sitl_window(start, stop, sc):
    '''
    SITL window information between `start` and `stop` and associated
    with spacecraft `sc`. Defines when the SITL can submit selections.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='BDM', event_type='sitl_window')


def _mission_data(start, stop, sc=None,
                  source=None, event_type=None):
    """
    Download MMS mission events. See the filters on the webpage
    for more ideas.
        https://lasp.colorado.edu/mms/sdc/public/about/events/#/

    NOTE: some sources, such as 'burst_segment', return a format
          that is not yet parsed properly.

    Parameters
    ----------
    start, stop : `datetime.datetime`, int
        Start and end of the data interval, specified as a time or
        orbit range.
    sc : str
        Spacecraft ID (mms, mms1, mms2, mms3, mms4) for which event
        information is to be returned.
    source : str
        Source of the mission event. Options include
            'Timeline', 'Burst', 'BDM', 'SITL'
    event_type : str
        Type of mission event. Options include
            BDM: sitl_window, evaluate_metadata, science_roi

    Returns
    -------
    data : dict
        Information about each event.
            start_time     - Start time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            end_time       - End time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            event_type     - Type of event
            sc_id          - Spacecraft to which the event applies
            source         - Source of event
            description    - Description of event
            discussion
            start_orbit    - Orbit on which the event started
            end_orbit      - Orbit on which the event ended
            tag
            id
            tstart         - Start time of event as datetime
            tend           - end time of event as datetime
    """
    url = 'https://lasp.colorado.edu/' \
          'mms/sdc/public/service/latis/mms_events_view.csv'

    start_date = None
    end_date = None
    start_orbit = None
    end_orbit = None

    # mission_events() returns numpy integers, so check for
    # those, too
    if isinstance(start, (int, np.integer)):
        start_orbit = start
    else:
        start_date = start
    if isinstance(stop, (int, np.integer)):
        end_orbit = stop
    else:
        end_date = stop

    query = {}
    if start_date is not None:
        query['start_time_utc>'] = start_date.strftime('%Y-%m-%d')
    if end_date is not None:
        query['end_time_utc<'] = end_date.strftime('%Y-%m-%d')

    if start_orbit is not None:
        query['start_orbit>'] = start_orbit
    if end_orbit is not None:
        query['end_orbit<'] = end_orbit

    if sc is not None:
        query['sc_id'] = sc
    if source is not None:
        query['source'] = source
    if event_type is not None:
        query['event_type'] = event_type

    resp = requests.get(url, params=query)
    data = _response_text_to_dict(resp.text)

    # Convert to useful types
    types = ['str', 'str', 'str', 'str', 'str', 'str', 'str',
             'int32', 'int32', 'str', 'int32']
    for items in zip(data, types):
        if items[1] == 'str':
            pass
        else:
            data[items[0]] = np.asarray(data[items[0]], dtype=items[1])

    # Add useful tags
    #   - Number of seconds elapsed
    #   - TAISTARTIME as datetime
    #   - TAIENDTIME as datetime

    # NOTE! If data['TAISTARTTIME'] is a scalar, this will not work
    #       unless everything after "in" is turned into a list
    data['tstart'] = [dt.datetime.strptime(
        value, '%Y-%m-%dT%H:%M:%S.%f'
    )
        for value in data['start_time']
    ]
    data['tend'] = [dt.datetime.strptime(
        value, '%Y-%m-%dT%H:%M:%S.%f'
    )
        for value in data['end_time']
    ]

    return data


def mission_events_v1(start_date=None, end_date=None,
                      start_orbit=None, end_orbit=None,
                      sc=None,
                      source=None, event_type=None):
    """
    Download MMS mission events. See the filters on the webpage
    for more ideas.
        https://lasp.colorado.edu/mms/sdc/public/about/events/#/

    NOTE: some sources, such as 'burst_segment', return a format
          that is not yet parsed properly.

    Parameters
    ----------
    start_date, end_date : `datetime.datetime`
        Start and end date of time interval. The interval is right-
        exclusive: [start_date, end_date). The time interval must
        encompass the desired data (e.g. orbit begin and end times)
        for it to be returned.
    start_orbit, end_orbit : `datetime.datetime`
        Start and end orbit of data interval. If provided with `start_date`
        or `end_date`, the two must overlap for any data to be returned.
    sc : str
        Spacecraft ID (mms, mms1, mms2, mms3, mms4) for which event
        information is to be returned.
    source : str
        Source of the mission event. Options include
            'Timeline', 'Burst', 'BDM', 'SITL'
    event_type : str
        Type of mission event. Options include
            BDM: sitl_window, evaluate_metadata, science_roi

    Returns
    -------
    data : dict
        Information about each event.
            start_time     - Start time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            end_time       - End time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            event_type     - Type of event
            sc_id          - Spacecraft to which the event applies
            source         - Source of event
            description    - Description of event
            discussion
            start_orbit    - Orbit on which the event started
            end_orbit      - Orbit on which the event ended
            tag
            id
            tstart         - Start time of event as datetime
            tend           - end time of event as datetime
    """
    url = 'https://lasp.colorado.edu/' \
          'mms/sdc/public/service/latis/mms_events_view.csv'

    query = {}
    if start_date is not None:
        query['start_time_utc>'] = start_date.strftime('%Y-%m-%d')
    if end_date is not None:
        query['end_time_utc<'] = end_date.strftime('%Y-%m-%d')

    if start_orbit is not None:
        query['start_orbit>'] = start_orbit
    if end_orbit is not None:
        query['end_orbit<'] = end_orbit

    if sc is not None:
        query['sc_id'] = sc
    if source is not None:
        query['source'] = source
    if event_type is not None:
        query['event_type'] = event_type

    resp = requests.get(url, params=query)
    data = _response_text_to_dict(resp.text)

    # Convert to useful types
    types = ['str', 'str', 'str', 'str', 'str', 'str', 'str',
             'int32', 'int32', 'str', 'int32']
    for items in zip(data, types):
        if items[1] == 'str':
            pass
        else:
            data[items[0]] = np.asarray(data[items[0]], dtype=items[1])

    # Add useful tags
    #   - Number of seconds elapsed
    #   - TAISTARTIME as datetime
    #   - TAIENDTIME as datetime
    #    data["start_time_utc"] = data.pop("start_time_utc "
    #                                      "(yyyy-mm-dd'T'hh:mm:ss.sss)"
    #                                      )
    #    data["end_time_utc"] = data.pop("end_time_utc "
    #                                    "(yyyy-mm-dd'T'hh:mm:ss.sss)"
    #                                    )

    # NOTE! If data['TAISTARTTIME'] is a scalar, this will not work
    #       unless everything after "in" is turned into a list
    data['tstart'] = [dt.datetime.strptime(
        value, '%Y-%m-%dT%H:%M:%S.%f'
    )
        for value in data['start_time']
    ]
    data['tend'] = [dt.datetime.strptime(
        value, '%Y-%m-%dT%H:%M:%S.%f'
    )
        for value in data['end_time']
    ]

    return data


def parse_file_name(fname):
    """
    Parse a file name compliant with MMS file name format guidelines.

    Parameters
    ----------
    fname : str
        File name to be parsed.

    Returns
    -------
    parts : tuple
        The tuple elements are:
            [0]: Spacecraft IDs
            [1]: Instrument IDs
            [2]: Data rate modes
            [3]: Data levels
            [4]: Optional descriptor (empty string if not present)
            [5]: Start times
            [6]: File version number
    """

    parts = os.path.basename(fname).split('_')

    # data_type = '*_selections'
    if 'selections' in fname:
        # datatype_glstype_YYYY-mm-dd-HH-MM-SS.sav
        if len(parts) == 3:
            gls_type = ''
        else:
            gls_type = parts[2]

        # (data_type, [gls_type,] start_date)
        out = ('_'.join(parts[0:2]), gls_type, parts[-1][0:-4])

    # data_type = 'science'
    else:
        # sc_instr_mode_level_[optdesc]_fstart_vVersion.cdf
        if len(parts) == 6:
            optdesc = ''
        else:
            optdesc = parts[4]

        # (sc, instr, mode, level, [optdesc,] start_date, version)
        out = (*parts[0:4], optdesc, parts[-2], parts[-1][1:-4])

    return out


def parse_time(times):
    """
    Parse the start time of MMS file names.

    Parameters
    ----------
    times : str, list
        Start times of file names.

    Returns
    -------
    parts : list
        A list of tuples. The tuple elements are:
            [0]: Year
            [1]: Month
            [2]: Day
            [3]: Hour
            [4]: Minute
            [5]: Second
    """
    if isinstance(times, str):
        times = [times]

    # Three types:
    #    srvy        YYYYMMDD
    #    brst        YYYYMMDDhhmmss
    #    selections  YYYY-MM-DD-hh-mm-ss
    parts = [None] * len(times)
    for idx, time in enumerate(times):
        if len(time) == 19:
            parts[idx] = (time[0:4], time[5:7], time[8:10],
                          time[11:13], time[14:16], time[17:]
                          )
        elif len(time) == 14:
            parts[idx] = (time[0:4], time[4:6], time[6:8],
                          time[8:10], time[10:12], time[12:14]
                          )
        else:
            parts[idx] = (time[0:4], time[4:6], time[6:8], '00', '00', '00')

    return parts


def read_eva_fom_structure(sav_filename):
    '''
    Returns a dictionary that mirrors the SITL selections fomstr structure
    that is in the IDL .sav file.

    Parameters
    ----------
    sav_filename : str
        Name of the IDL sav file containing the SITL selections

    Returns
    -------
    data : dict
        The FOM structure.
            valid                    : 1 if the fom structure is valid, 0 otherwise
            error                    : Error string for invalid fom structures
            algversion
            sourceid                 : username of the SITL that made the selections
            cyclestart
            numcycles
            nsegs                    : number of burst segments
            start                    : index into timestamps of start time for each burst segment
            stop                     : index into timestamps of stop time for each burst segment
            seglengths
            fom                      : figure of merit for each burst segment
            nubffs
            mdq                      : mission data quality
            timestamps               : timestamp (TAI seconds since 1958) of each mdq
            targetbuffs
            fomave
            targetratio
            minsegmentsize
            maxsegmentsize
            pad
            searchratio
            fomwindowsize
            fomslope
            fomskew
            fombias
            metadatainfo
            oldestavailableburstdata :
            metadataevaltime
            discussion               : description of each burst segment given by the SITL
            note                     : note given by SITL to data within SITL window
            datetimestamps           : timestamps converted to datetimes
            start_time               : start time of the burst segment
            end_time                 : end time of the burst segment
            tstart                   : datetime timestamp of the start of each burst segment
            tstop                    : datetime timestamp of the end of each burst segment
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sav = readsav(sav_filename)

    assert 'fomstr' in sav, 'save file does not have a fomstr structure'
    fomstr = sav['fomstr']

    # Handle invalid structures
    #   - example: abs_selections_2017-10-29-09-25-34.sav
    if fomstr.valid[0] == 0:
        d = {'valid': int(fomstr.valid[0]),
             'error': fomstr.error[0].decode('utf-8'),
             'errno': int(fomstr.errno[0])
             }
        return d

    d = {'valid': int(fomstr.valid[0]),
         'error': fomstr.error[0],
         'algversion': fomstr.algversion[0].decode('utf-8'),
         'sourceid': [x.decode('utf-8') for x in fomstr.sourceid[0]],
         'cyclestart': int(fomstr.cyclestart[0]),
         'numcycles': int(fomstr.numcycles[0]),
         'nsegs': int(fomstr.nsegs[0]),
         'start': fomstr.start[0].tolist(),
         'stop': fomstr.stop[0].tolist(),
         'seglengths': fomstr.seglengths[0].tolist(),
         'fom': fomstr.fom[0].tolist(),
         'nbuffs': int(fomstr.nbuffs[0]),
         'mdq': fomstr.mdq[0].tolist(),
         'timestamps': fomstr.timestamps[0].tolist(),
         'targetbuffs': int(fomstr.targetbuffs[0]),
         'fomave': float(fomstr.fomave[0]),
         'targetratio': float(fomstr.targetratio[0]),
         'minsegmentsize': float(fomstr.minsegmentsize[0]),
         'maxsegmentsize': float(fomstr.maxsegmentsize[0]),
         'pad': int(fomstr.pad[0]),
         'searchratio': float(fomstr.searchratio[0]),
         'fomwindowsize': int(fomstr.fomwindowsize[0]),
         'fomslope': float(fomstr.fomslope[0]),
         'fomskew': float(fomstr.fomskew[0]),
         'fombias': float(fomstr.fombias[0]),
         'metadatainfo': fomstr.metadatainfo[0].decode('utf-8'),
         'oldestavailableburstdata': fomstr.oldestavailableburstdata[0].decode('utf-8'),
         'metadataevaltime': fomstr.metadataevaltime[0].decode('utf-8')
         }
    try:
        d['discussion'] = [x.decode('utf-8') for x in fomstr.discussion[0]]
    except AttributeError:
        d['discussion'] = ['ABS Selections'] * len(d['start'])
    try:
        d['note'] = fomstr.note[0].decode('utf-8')
    except AttributeError:
        d['note'] = 'ABS Selections'

    # Convert TAI to datetime
    #   - timestaps are TAI seconds elapsed since 1958-01-01
    #   - tt2000 are nanoseconds elapsed since 2000-01-01
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    tepoch = epochs.CDFepoch()
    d['datetimestamps'] = tepoch.to_datetime(
        np.asarray(d['timestamps']) * int(1e9) +
        t_1958
    )

    # FOM structure (copy procedure from IDL/SPEDAS/EVA)
    #   - eva_sitl_load_soca_simple
    #   - eva_sitl_strct_read
    #   - mms_convert_from_tai2unix
    #   - mms_tai2unix
    if 'fomslope' in d:
        if d['stop'][d['nsegs'] - 1] >= d['numcycles']:
            raise ValueError('Number of segments should be <= # cycles.')

        taistarttime = []
        taiendtime = []
        tstart = []
        tstop = []
        t_fom = [d['datetimestamps'][0]]
        fom = [0]
        dtai_last = (d['timestamps'][d['numcycles'] - 1] -
                     d['timestamps'][d['numcycles'] - 2])
        dt_last = (d['datetimestamps'][d['numcycles'] - 1] -
                   d['datetimestamps'][d['numcycles'] - 2])

        # Extract the start and stop times of the FOM values
        # Create a time series for FOM values
        for idx in range(d['nsegs']):
            taistarttime.append(d['timestamps'][d['start'][idx]])
            tstart.append(d['datetimestamps'][d['start'][idx]])
            if d['stop'][idx] <= d['numcycles'] - 1:
                taiendtime.append(d['timestamps'][d['stop'][idx] + 1])
                tstop.append(d['datetimestamps'][d['stop'][idx] + 1])
            else:
                taiendtime.append(d['timestamps'][d['numcycles'] - 1] + dtai_last)
                tstop.append(d['datetimestamps'][d['numcycles'] - 1] + dt_last)

        # Append the last time stamp to the time series
        t_fom.append(d['datetimestamps'][d['numcycles'] - 1] + dt_last)
        fom.append(0)

    # BAK structure
    else:
        raise NotImplemented('BAK structure has not been implemented')
        nsegs = len(d['fom'])  # BAK

    # Add to output structure
    d['taistarttime'] = taistarttime
    d['taiendtime'] = taiendtime
    d['start_time'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in tstart]
    d['stop_time'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in tstop]
    d['tstart'] = tstart
    d['tstop'] = tstop
    d['createtime'] = [file_start_time(sav_filename)] * d['nsegs']

    return d


def read_gls_csv(filename):
    """
    Read a ground loop selections (gls) CSV file.

    Parameters
    ----------
    filename : str
        Name of the CSV file to be read

    Returns
    -------
    data : dict
        Data contained in the CSV file
    """
    # Dictionary to hold data from csv file
    keys = ['start_time', 'stop_time', 'sourceid', 'fom', 'discussion',
            'taistarttime', 'taiendtime', 'tstart', 'tstop', 'createtime']
    data = {key: [] for key in keys}

    # CSV files have their generation time in the file name.
    # Multiple CSV files may have been created for the same
    # data interval, which results in duplicate data. Use
    # a set to keep only unique data entries.
    tset = set()
    nold = 0

    # Constant for converting times to TAI seconds since 1958
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])

    # Parse each row of all files
    skip_file = False
    nentry_skip = 0
    nentry_expand = 0
    with open(filename) as f:
        fstart = file_start_time(filename)

        reader = csv.reader(f)
        for row in reader:
            tstart = dt.datetime.strptime(
                row[0], '%Y-%m-%d %H:%M:%S'
            )
            tstop = dt.datetime.strptime(
                row[1], '%Y-%m-%d %H:%M:%S'
            )

            # Convert times to TAI seconds since 1958
            t0 = _datetime_to_list(tstart)
            t1 = _datetime_to_list(tstop)
            t0 = int((epochs.CDFepoch.compute_tt2000(t0) - t_1958) // 1e9)
            t1 = int((epochs.CDFepoch.compute_tt2000(t1) - t_1958) // 1e9)

            # Ensure selections have a minimum length of 10 seconds
            if (t1 - t0) == 0:
                t1 += int(10)
                tstop += dt.timedelta(seconds=10)
                row[1] = dt.datetime.strftime(
                    tstop, '%Y-%m-%d %H:%M:%S'
                )
                nentry_expand += 1

            # Some burst segments are unrealistically long
            #   - Usually, the longest have one selection per file
            if ((t1 - t0) > 3600):
                with open(filename) as f_test:
                    nrows = sum(1 for row in f_test)
                if nrows == 1:
                    skip_file = True
                    break

            # Some entries have negative durations
            if (t1 - t0) < 0:
                nentry_skip += 1
                continue

            # Store data
            data['taistarttime'].append(t0)
            data['taiendtime'].append(t1)
            data['start_time'].append(row[0])
            data['stop_time'].append(row[1])
            data['fom'].append(float(row[2]))
            data['discussion'].append(','.join(row[3:]))
            data['tstart'].append(tstart)
            data['tstop'].append(tstop)
            data['createtime'].append(fstart)

        # Source ID is the name of the GLS model
        parts = parse_file_name(filename)
        data['sourceid'].extend([parts[1]] * len(data['fom']))

        # Errors
        data['errors'] = {'fskip': skip_file,
                          'nexpand': nentry_expand,
                          'nskip': nentry_skip
                          }

    return data


def _sdc_parse_form(r):
    '''Parse key-value pairs from the log-in form

    Parameters
    ----------
    r (object):    requests.response object.

    Returns
    -------
    form (dict):   key-value pairs parsed from the form.
    '''
    # Find action URL
    pstart = r.text.find('<form')
    pend = r.text.find('>', pstart)
    paction = r.text.find('action', pstart, pend)
    pquote1 = r.text.find('"', pstart, pend)
    pquote2 = r.text.find('"', pquote1 + 1, pend)
    url5 = r.text[pquote1 + 1:pquote2]
    url5 = url5.replace('&#x3a;', ':')
    url5 = url5.replace('&#x2f;', '/')

    # Parse values from the form
    pinput = r.text.find('<input', pend + 1)
    inputs = {}
    while pinput != -1:
        # Parse the name-value pair
        pend = r.text.find('/>', pinput)

        # Name
        pname = r.text.find('name', pinput, pend)
        pquote1 = r.text.find('"', pname, pend)
        pquote2 = r.text.find('"', pquote1 + 1, pend)
        name = r.text[pquote1 + 1:pquote2]

        # Value
        if pname != -1:
            pvalue = r.text.find('value', pquote2 + 1, pend)
            pquote1 = r.text.find('"', pvalue, pend)
            pquote2 = r.text.find('"', pquote1 + 1, pend)
            value = r.text[pquote1 + 1:pquote2]
            value = value.replace('&#x3a;', ':')

            # Extract the values
            inputs[name] = value

        # Next iteraction
        pinput = r.text.find('<input', pend + 1)

    form = {'url': url5,
            'payload': inputs}

    return form


def sdc_login(username):
    '''
    Log-In to the MMS Science Data Center.

    Parameters:
    -----------
    username : str
        Account username.
    password : str
        Account password.

    Returns:
    --------
    Cookies : dict
        Session cookies for continued access to the SDC. Can
        be passed to an instance of requests.Session.
    '''

    # Ask for the password
    password = getpass()

    # Disable warnings because we are not going to obtain certificates
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Attempt to access the site
    #   - Each of the redirects are stored in the history attribute
    url0 = 'https://lasp.colorado.edu/mms/sdc/team/'
    r = requests.get(url0, verify=False)

    # Extract cookies and url
    cookies = r.cookies
    for response in r.history:
        cookies.update(response.cookies)

        try:
            url = response.headers['Location']
        except:
            pass

    # Submit login information
    payload = {'j_username': username, 'j_password': password}
    r = requests.post(url, cookies=cookies, data=payload, verify=False)

    # After submitting info, we land on a page with a form
    #   - Parse form and submit values to continue
    form = _sdc_parse_form(r)
    r = requests.post(form['url'],
                      cookies=cookies,
                      data=form['payload'],
                      verify=False
                      )

    # Update cookies to get session information
    #    cookies = r.cookies
    for response in r.history:
        cookies.update(response.cookies)

    return cookies

def time_to_orbit(time, sc='mms1', delta=10):
    '''
    Identify the orbit in which a time falls.

    Parameters
    ----------
    time : `datetime.datetime`
        Time within the orbit
    sc : str
        Spacecraft identifier
    delta : int
        Number of days around around the time of interest in
        which to search for the orbit. Should be the duration
        of at least one orbit.

    Returns
    -------
    orbit : int
        Orbit during which `time` occurs
    '''
    # sdc.mission_events filters by date, and the dates are right-exclusive:
    # [tstart, tstop). For it to return data on the date of `time`, `time`
    # must be rounded up to the next day. Start the time interval greater
    # than one orbit prior than the start time. The desired orbit should then
    # be the last orbit in the list
    tstop = dt.datetime.combine(time.date() + dt.timedelta(days=delta),
                                dt.time(0, 0, 0))
    tstart = tstop - dt.timedelta(days=2 * delta)
    orbits = mission_events('orbit', tstart, tstop, sc=sc)

    orbit = None
    for idx in range(len(orbits['tstart'])):
        if (time > orbits['tstart'][idx]) and (time < orbits['tend'][idx]):
            orbit = orbits['start_orbit'][idx]
    if orbit is None:
        ValueError('Did not find correct orbit!')

    return orbit


if __name__ == '__main__':
    '''Download data'''

    # Inputs common to each calling sequence
    sc = sys.argv[0]
    instr = sys.argv[1]
    mode = sys.argv[2]
    level = sys.argv[3]

    # Basic dataset
    if len(sys.argv) == 7:
        optdesc = None
        start_date = sys.argv[4]
        end_date = sys.argv[5]

    # Optional descriptor given
    elif len(sys.argv) == 8:
        optdesc = sys.argv[4]
        start_date = sys.argv[5]
        end_date = sys.argv[6]

    # Error
    else:
        raise TypeError('Incorrect number if inputs.')

    # Create the request
    api = MrMMS_SDC_API(sc, instr, mode, level,
                        optdesc=optdesc, start_date=start_date, end_date=end_date)

    # Download the data
    files = api.download_files()

def sort_files(files):
    """
    Sort MMS file names by data product and time.

    Parameters:
    files : str, list
        Files to be sorted

    Returns
    -------
    sorted : tuple
        Sorted file names. Each tuple element corresponds to
        a unique data product.
    """

    # File types and start times
    parts = [parse_file_name(file) for file in files]
    bases = ['_'.join(p[0:5]) for p in parts]
    tstart = [p[-2] for p in parts]

    # Sort everything
    idx = sorted(range(len(tstart)), key=lambda k: tstart[k])
    bases = [bases[i] for i in idx]
    files = [files[i] for i in idx]

    # Find unique file types
    fsort = []
    uniq_bases = list(set(bases))
    for ub in uniq_bases:
        fsort.append([files[i] for i, b in enumerate(bases) if b == ub])

    return tuple(fsort)




# the following is the code from the util.py file


# cdfepoch requires datetimes to be broken down into 9-element lists
def datetime_to_list(t):
    return [t.year, t.month, t.day,
            t.hour, t.minute, t.second,
            int(t.microsecond // 1e3),
            int(t.microsecond % 1e3),
            0]


def tt2000_range(cdf, t_vname, start_date, end_date):
    # Create lists
    tstart = datetime_to_list(start_date)
    tend = datetime_to_list(end_date)

    # Convert to TT2000
    tstart = cdflib.cdfepoch.compute(tstart)
    tend = cdflib.cdfepoch.compute(tend)

    # Find the time range
    return cdf.epochrange(epoch=t_vname, starttime=tstart, endtime=tend)


def from_cdflib(files, varname, start_date, end_date):
    global cdf_vars
    global file_vars

    if isinstance(files, str):
        files = [files]
    tstart = datetime_to_list(start_date)
    tend = datetime_to_list(end_date)

    # Extract metadata
    cdf_vars = {}
    for file in files:
        file_vars = {}
        cdf = cdflib.CDF(file)

        try:
            data = cdflib_readvar(cdf, varname, tstart, tend)
        except:
            cdf.close()
            raise

        cdf.close()

    return data


def cdflib_readvar(cdf, varname, tstart, tend):
    global cdf_vars
    global file_vars

    # Data has already been read from this file
    if varname in file_vars:
        var = file_vars[varname]
    else:
        time_types = ('CDF_EPOCH', 'CDF_EPOCH16', 'CDF_TIME_TT2000')
        varinq = cdf.varinq(varname)

        # Convert epochs to datetimes
        data = cdf.varget(variable=varname, starttime=tstart, endtime=tend)
        if varinq['Data_Type_Description'] in time_types:
            data = cdflib.cdfepoch().to_datetime(data)

        # If the variable has been read from a different file, append
        if (varname in cdf_vars) and varinq['Rec_Vary']:
            d0 = cdf_vars[varname]
            data = np.append(d0['data'], data, 0)

        # Create the variable
        var = {'name': varname,
               'data': data,
               'rec_vary': varinq['Rec_Vary'],
               'cdf_name': varinq['Variable'],
               'cdf_type': varinq['Data_Type_Description']
               }

        # List as read
        #  - Prevent infinite loop. Must save the variable in the registry
        #  so that variable attributes do not try to read the same variable
        #  again.
        cdf_vars[varname] = var
        file_vars[varname] = var

        # Read the metadata
        cdflib_attget(cdf, var, tstart, tend)

    return var


def cdflib_attget(cdf, var, tstart, tend):
    # Get variable attributes for given variable
    varatts = cdf.varattsget(var['cdf_name'])

    # Get names of all cdf variables
    cdf_varnames = cdf.cdf_info()['zVariables']

    # Follow pointers to retrieve data
    for attrname, attrvalue in varatts.items():
        var[attrname] = attrvalue
        if isinstance(attrvalue, str) and (attrvalue in cdf_varnames):
            var[attrvalue] = cdflib_readvar(cdf, attrvalue, tstart, tend)


def plot_1D(data, axes):
    # Plot the data
    lines = axes.plot(mdates.date2num(data[data['DEPEND_0']]['data']),
                      data['data'])

    try:
        axes.set_yscale(data['SCALETYP'])
    except KeyError:
        pass

    try:
        for line, color in zip(lines, data['color']):
            line.set_color(color)
    except KeyError:
        pass

    try:
        # Set the label for each line so that they can
        # be returned by Legend.get_legend_handles_labels()
        for line, label in zip(lines, data[data['LABL_PTR_1']]['data']):
            line.set_label(label)

        # Create the legend outside the right-most axes
        leg = axes.legend(bbox_to_anchor=(1.05, 1),
                          borderaxespad=0.0,
                          frameon=False,
                          handlelength=0,
                          handletextpad=0,
                          loc='upper left')

        # Color the text the same as the lines
        for line, text in zip(lines, leg.get_texts()):
            text.set_color(line.get_color())

    except KeyError:
        pass


def plot_2D(data, axes):
    # Convert time to seconds and reshape to 2D arrays
    x0 = mdates.date2num(data[data['DEPEND_0']]['data'])
    x1 = data[data['DEPEND_1']]['data']
    if x0.ndim == 1:
        x0 = np.repeat(x0[:, np.newaxis], data['data'].shape[1], axis=1)
    if x1.ndim == 1:
        x1 = np.repeat(x1[np.newaxis, :], data['data'].shape[0], axis=0)

    # Format the image
    y = data['data'][0:-1, 0:-1]
    try:
        if data['SCALETYP'] == 'log':
            y = np.ma.log(y)
    except KeyError:
        pass

    # Create the image
    im = axes.pcolorfast(x0, x1, y, cmap='nipy_spectral')
    axes.images.append(im)

    try:
        axes.set_yscale(data[data['DEPEND_1']]['SCALETYP'])
    except KeyError:
        pass

    # Create a colorbar to the right of the image
    cbaxes = inset_axes(axes,
                        width='1%', height='100%', loc=4,
                        bbox_to_anchor=(0, 0, 1.05, 1),
                        bbox_transform=axes.transAxes,
                        borderpad=0)
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')


def plot_burst_selections(sc, start_date, end_date,
                          figsize=(5.5, 7)):
    mode = 'srvy'
    level = 'l2'




    # FGM
    b_vname = '_'.join((sc, 'fgm', 'b', 'gse', mode, level))
    mms = MrMMS_SDC_API(sc, 'fgm', mode, level,
                            start_date=start_date, end_date=end_date)
    files = mms.download_files()
    files = sort_files(files)[0]
#us different function here
    fgm_data = ff.cdf_to_df(files,b_vname) # from_cdflib(files, b_vname,
                           #start_date, end_date)
    fgm_data['data'] = fgm_data['data'][:, [3, 0, 1, 2]]
    fgm_data['color'] = ['Black', 'Blue', 'Green', 'Red']
    fgm_data[fgm_data['LABL_PTR_1']]['data'] = ['|B|', 'Bx', 'By', 'Bz']

    # FPI DIS
    fpi_mode = 'fast'
    ni_vname = '_'.join((sc, 'dis', 'numberdensity', fpi_mode))
    espec_i_vname = '_'.join((sc, 'dis', 'energyspectr', 'omni', fpi_mode))
    BV_vname = '_'.join((sc,'dis','bulkv_gse',fpi_mode))
    mms = MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
                            optdesc='dis-moms',
                            start_date=start_date, end_date=end_date)
    files = mms.download_files()
    files = sort_files(files)[0]

    ni_data = ff.cdf_to_df(files,ni_vname)#from_cdflib(files, ni_vname,
                  #        start_date, end_date)
    especi_data = ff.cdf_to_df(files,espec_i_vname)#from_cdflib(files, espec_i_vname,
                   #           start_date, end_date)
    BV_data = ff.cdf_to_df(files,BV_name)#from_cdflib(files, BV_vname,
               #           start_date,end_date)

    # FPI DES
    ne_vname = '_'.join((sc, 'des', 'numberdensity', fpi_mode))
    espec_e_vname = '_'.join((sc, 'des', 'energyspectr', 'omni', fpi_mode))
    mms = MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
                            optdesc='des-moms',
                            start_date=start_date, end_date=end_date)
    files = mms.download_files()
    files = sort_files(files)[0]
    #ne_data = from_cdflib(files, ne_vname,
    #                      start_date, end_date)
    espece_data = ff.cdf_to_df(files,espec_e_vname)#from_cdflib(files, espec_e_vname,
                              # start_date, end_date)

    # Grab selections
    abs_data = selections('abs', start_date, end_date,
                              combine=True, sort=True,filter= 'MP')
    sitl_data = selections('sitl+back', start_date, end_date,
                               combine=True, sort=True,filter='MP')
    gls_data = selections('mp-dl-unh', start_date, end_date,
                              combine=True, sort=True,filter='MP')

    # SITL data time series
    t_abs = []
    x_abs = []
    for selection in abs_data:
        t_abs.extend([selection.tstart, selection.tstart,
                      selection.tstop, selection.tstop])
        x_abs.extend([0, selection.fom, selection.fom, 0])
    if len(abs_data) == 0:
        t_abs = [start_date, end_date]
        x_abs = [0, 0]
    abs = {'data': x_abs,
           'DEPEND_0': 't',
           't': {'data': t_abs}}

    t_sitl = []
    x_sitl = []
    for selection in sitl_data:
        t_sitl.extend([selection.tstart, selection.tstart,
                       selection.tstop, selection.tstop])
        x_sitl.extend([0, selection.fom, selection.fom, 0])
    if len(sitl_data) == 0:
        t_sitl = [start_date, end_date]
        x_sitl = [0, 0]
    sitl = {'data': x_sitl,
            'DEPEND_0': 't',
            't': {'data': t_sitl}}

    t_gls = []
    x_gls = []
    for selection in gls_data:
        t_gls.extend([selection.tstart, selection.tstart,
                      selection.tstop, selection.tstop])
        x_gls.extend([0, selection.fom, selection.fom, 0])
    if len(gls_data) == 0:
        t_gls = [start_date, end_date]
        x_gls = [0, 0]
    gls = {'data': x_gls,
           'DEPEND_0': 't',
           't': {'data': t_gls}}

    # Setup plot
    nrows = 8#7
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, squeeze=False)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    # Plot FGM
    plot_2D(especi_data, axes[0, 0])
    axes[0, 0].set_title(sc.upper())
    fig.axes[-1].set_label('DEF')
    axes[0, 0].set_ylabel('$E_{ion}$\n(eV)')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_xlabel('')

    plot_2D(espece_data, axes[1, 0])
    fig.axes[-1].set_label('DEF\nLog_{10}(keV/(cm^2 s sr keV))')
    axes[1, 0].set_ylabel('$E_{e-}$\n(eV)')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_title('')

    plot_1D(fgm_data, axes[2, 0])
    axes[2, 0].set_ylabel('B\n(nT)')
    axes[2, 0].set_xticks([])
    axes[2, 0].set_xlabel('')
    axes[2, 0].set_title('')

    plot_1D(ni_data, axes[3, 0])
    axes[3, 0].set_ylabel('$N_{i}$\n($cm^{-3}$)')
    axes[3, 0].set_xticks([])
    axes[3, 0].set_xlabel('')
    axes[3, 0].set_title('')

    plot_1D(BV_data, axes[4, 0])
    axes[4, 0].set_ylabel('V\n(km/s)')
    # axes[4, 0].set_ylim(200, 200)
    axes[4, 0].set_xticks([])
    axes[4, 0].set_xlabel('')
    axes[4, 0].set_title('')

    plot_1D(abs, axes[5, 0])
    axes[5, 0].set_ylabel('ABS')
    axes[5, 0].set_xticks([])
    axes[5, 0].set_xlabel('')
    axes[5, 0].set_title('')

    plot_1D(gls, axes[6, 0])
    axes[6, 0].set_ylabel('GLS')
    axes[6, 0].set_ylim(0, 200)
    axes[6, 0].set_xticks([])
    axes[6, 0].set_xlabel('')
    axes[6, 0].set_title('')

    plot_1D(sitl, axes[7, 0])
    axes[7, 0].set_ylabel('SITL')
    axes[7, 0].set_title('')
    axes[7, 0].xaxis.set_major_locator(locator)
    axes[7, 0].xaxis.set_major_formatter(formatter)
    for tick in axes[7, 0].get_xticklabels():
        tick.set_rotation(45)

    # Set a common time range
    plt.setp(axes, xlim=mdates.date2num([start_date, end_date]))
    plt.subplots_adjust(left=0.27, right=0.85, top=0.93)
    return fig, axes
()