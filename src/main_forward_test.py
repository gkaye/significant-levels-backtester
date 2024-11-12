import pickle
import time

import pandas

from directory_definitions import *
from src.utilities import compute_trades_daterange, compute_all_parameter_stats, compute_derivatives, populate_derivatives

# CONFIGURATION
OUTPUT_DIR = f'{OUTPUT_DIR}_ft'
START_DATE = '2022-12-01'
END_DATE = '2023-02-01'

BARS_FILE_PATH = f'{DOWNLOADS_DIR}/bars/SPY_2020-06-14_to_2023-02-16.pickle'

# 142506
LONG_PARAMETER_CONFIGURATION = {
    'min_distance_start': 0.75,
    'min_distance_end': 0.75,
    'min_distance_step': 1.0,
    'max_rvol_start': 1.5,
    'max_rvol_end': 1.5,
    'max_rvol_step': 1.0,
    'stop_start': 0.2,
    'stop_end': 0.2,
    'stop_step': 1.0,
    'buffer_start': 0.4,
    'buffer_end': 0.4,
    'buffer_step': 1.0,
    'target_start': 0.2,
    'target_end': 0.2,
    'target_step': 1.0,
}

# 401078
SHORT_PARAMETER_CONFIGURATION = {
    'min_distance_start': 1.75,
    'min_distance_end': 1.75,
    'min_distance_step': 1.00,
    'max_rvol_start': 1.4,
    'max_rvol_end': 1.4,
    'max_rvol_step': 1.0,
    'stop_start': 0.8,
    'stop_end': 0.8,
    'stop_step': 1.0,
    'buffer_start': 0.2,
    'buffer_end': 0.2,
    'buffer_step': 1.0,
    'target_start': 0.4,
    'target_end': 0.4,
    'target_step': 1.0,
}
# END CONFIGURATION


def wrapped_compute_trades_daterange(start_date, end_date, parameter_configuration, function_name='compute_trades_daterange()', use_cache=False, cache_directory=f'{OUTPUT_DIR}/trades.pickle'):
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


def wrapped_compute_all_parameter_stats(all_trades, parameter_configuration, function_name='compute_all_parameter_stats()', use_cache=False, cache_directory=f'{OUTPUT_DIR}/stats.pickle'):
    if use_cache:
        print(f'Retrieved {function_name} from cache')
        return pandas.read_pickle(cache_directory)

    print(f'Starting {function_name}')
    start = time.time()
    ret = compute_all_parameter_stats(all_trades, parameter_configuration)
    print(f'{function_name}: {(time.time() - start):.1f} seconds elapsed')
    print()
    with open(cache_directory, 'wb') as handle:
        pickle.dump(ret, handle)

    return ret


def wrapped_compute_derivatives(bars_file_path, function_name='compute_derivatives()', use_cache=False, cache_directory=f'{OUTPUT_DIR}/bars_with_derivatives.pickle'):
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
    # Compute bars with RVOL
    bars_with_derivatives = wrapped_compute_derivatives(BARS_FILE_PATH, use_cache=True)

    # Get trades
    long_trades = wrapped_compute_trades_daterange(START_DATE, END_DATE, LONG_PARAMETER_CONFIGURATION, use_cache=False, cache_directory=f'{OUTPUT_DIR}/trades_longs_ft.pickle')
    short_trades = wrapped_compute_trades_daterange(START_DATE, END_DATE, SHORT_PARAMETER_CONFIGURATION, use_cache=False, cache_directory=f'{OUTPUT_DIR}/trades_shorts_ft.pickle')

    # Downselect
    long_trades = long_trades[long_trades['direction'] == 'long']
    short_trades = short_trades[short_trades['direction'] == 'short']

    # Add derivatives to trades
    long_trades = wrapped_populate_derivatives(long_trades, bars_with_derivatives, use_cache=False, cache_directory=f'{OUTPUT_DIR}/trades_longs_ft_with_rvol.pickle')
    short_trades = wrapped_populate_derivatives(short_trades, bars_with_derivatives, use_cache=False, cache_directory=f'{OUTPUT_DIR}/trades_shorts_ft_with_rvol.pickle')

    # Compute stats
    long_stats = wrapped_compute_all_parameter_stats(long_trades, LONG_PARAMETER_CONFIGURATION, function_name='compute_all_parameter_stats(longs)', cache_directory=f'{OUTPUT_DIR}/stats_longs_ft.pickle')
    short_stats = wrapped_compute_all_parameter_stats(short_trades, SHORT_PARAMETER_CONFIGURATION, function_name='compute_all_parameter_stats(shorts)', cache_directory=f'{OUTPUT_DIR}/stats_shorts_ft.pickle')
