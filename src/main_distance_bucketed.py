import pickle
import time

import pandas

from directory_definitions import *
from src.utilities import compute_trades_daterange, compute_derivatives, populate_derivatives, \
    compute_all_parameter_stats_distance_bucketed

# CONFIGURATION
OUTPUT_DIR = f'{OUTPUT_DIR}_bucketed'

START_DATE = '2021-11-30'
END_DATE = '2022-11-30'

BARS_FILE_PATH = f'{DOWNLOADS_DIR}/bars/SPY_2020-06-14_to_2023-02-16.pickle'

PARAMETER_CONFIGURATION_FOR_TRADES = {
    'stop_start': 0.2,
    'stop_end': 1.0,
    'stop_step': 0.1,
    'buffer_start': 0.0,
    'buffer_end': 0.5,
    'buffer_step': 0.1,
    'target_start': 0.2,
    'target_end': 4,
    'target_step': 0.1,
}
PARAMETER_CONFIGURATION_LONGS = {
    'distance_bucket_points': [0, 0.5, 1.25, 2.0, 10],
    'max_rvol_start': 1.0,
    'max_rvol_end': 4,
    'max_rvol_step': 0.1,
    'stop_start': 0.2,
    'stop_end': 1.0,
    'stop_step': 0.1,
    'buffer_start': 0.0,
    'buffer_end': 0.5,
    'buffer_step': 0.1,
    'target_start': 0.2,
    'target_end': 4,
    'target_step': 0.1,
}
PARAMETER_CONFIGURATION_SHORTS = {
    'distance_bucket_points': [0, 0.25, 1.00, 1.75, 10],
    'max_rvol_start': 1.0,
    'max_rvol_end': 4,
    'max_rvol_step': 0.1,
    'stop_start': 0.2,
    'stop_end': 1.0,
    'stop_step': 0.1,
    'buffer_start': 0.0,
    'buffer_end': 0.5,
    'buffer_step': 0.1,
    'target_start': 0.2,
    'target_end': 4,
    'target_step': 0.1,
}
# END CONFIGURATION


def wrapped_compute_trades_daterange(start_date, end_date, parameter_configuration,
                                     function_name='compute_trades_daterange()', use_cache=False,
                                     cache_directory=f'{OUTPUT_DIR}/trades.pickle'):
    if use_cache:
        print(f'Retrieved {function_name} from cache')
        return pandas.read_pickle(cache_directory)

    print(f'Starting {function_name}')
    start = time.time()
    ret = compute_trades_daterange(start_date, end_date, parameter_configuration)
    print(f'{function_name}: {(time.time() - start):.1f} seconds elapsed')
    print()
    with open(cache_directory, 'wb') as handle:
        pickle.dump(ret, handle)

    return ret


def wrapped_compute_all_parameter_stats_distance_bucketed(all_trades, parameter_configuration,
                                                          function_name='compute_all_parameter_stats()',
                                                          use_cache=False,
                                                          cache_directory=f'{OUTPUT_DIR}/stats.pickle'):
    if use_cache:
        print(f'Retrieved {function_name} from cache')
        return pandas.read_pickle(cache_directory)

    print(f'Starting {function_name}')
    start = time.time()
    ret = compute_all_parameter_stats_distance_bucketed(all_trades, parameter_configuration)
    print(f'{function_name}: {(time.time() - start):.1f} seconds elapsed')
    print()
    with open(cache_directory, 'wb') as handle:
        pickle.dump(ret, handle)

    return ret


def wrapped_compute_derivatives(bars_file_path, function_name='compute_derivatives()', use_cache=False,
                         cache_directory=f'{OUTPUT_DIR}/bars_with_derivatives.pickle'):
    if use_cache:
        print(f'Retrieved {function_name} from cache')
        return pandas.read_pickle(cache_directory)

    print(f'Starting {function_name}')
    start = time.time()
    ret = compute_derivatives(bars_file_path)
    print(f'{function_name}: {(time.time() - start):.1f} seconds elapsed')
    print()
    with open(cache_directory, 'wb') as handle:
        pickle.dump(ret, handle)

    return ret


def wrapped_populate_derivatives(trades, bars_with_derivatives, function_name='populate_derivatives()', use_cache=False,
                          cache_directory=f'{OUTPUT_DIR}/trades_with_derivatives.pickle'):
    if use_cache:
        print(f'Retrieved {function_name} from cache')
        return pandas.read_pickle(cache_directory)

    print(f'Starting {function_name}')
    start = time.time()
    ret = populate_derivatives(trades, bars_with_derivatives)
    print(f'{function_name}: {(time.time() - start):.1f} seconds elapsed')
    print()
    with open(cache_directory, 'wb') as handle:
        pickle.dump(ret, handle)

    return ret


if __name__ == '__main__':
    # Get trades
    trades = wrapped_compute_trades_daterange(START_DATE, END_DATE, PARAMETER_CONFIGURATION_FOR_TRADES, use_cache=False)

    # Compute bars with RVOL
    bars_with_derivatives = wrapped_compute_derivatives(BARS_FILE_PATH, use_cache=True)

    # Add derivatives to trades
    trades = wrapped_populate_derivatives(trades, bars_with_derivatives, use_cache=False)

    # Compute stats for short trades
    short_trades = trades[trades['direction'] == 'short']
    short_stats = wrapped_compute_all_parameter_stats_distance_bucketed(short_trades, PARAMETER_CONFIGURATION_SHORTS,
                                                      function_name='compute_all_parameter_stats(shorts)',
                                                      cache_directory=f'{OUTPUT_DIR}/stats_shorts.pickle')

    # Compute stats for long trades
    long_trades = trades[trades['direction'] == 'long']
    long_stats = wrapped_compute_all_parameter_stats_distance_bucketed(long_trades, PARAMETER_CONFIGURATION_LONGS,
                                                     function_name='compute_all_parameter_stats(longs)',
                                                     cache_directory=f'{OUTPUT_DIR}/stats_longs.pickle')
