import pickle
import time

import pandas

from directory_definitions import *
from src.utilities import compute_trades_daterange, compute_all_parameter_stats, compute_derivatives, populate_derivatives

# CONFIGURATION
START_DATE = '2022-11-01'
END_DATE = '2023-02-16'

BARS_FILE_PATH = f'{DOWNLOADS_DIR}/bars/SPY_2020-06-14_to_2023-02-16.pickle'

PARAMETER_CONFIGURATION = {
    'min_distance_start': 1.7,
    'min_distance_end': 1.7,
    'min_distance_step': 0.1,
    'max_rvol_start': 2.1,
    'max_rvol_end': 2.1,
    'max_rvol_step': 0.1,
    'buffer_start': 0.3,
    'buffer_end': 0.3,
    'buffer_step': 0.1,
    'stop_a_start': 0,
    'stop_a_end': 0.0,
    'stop_a_step': 0.2,
    'stop_b_start': 2.1,
    'stop_b_end': 2.1,
    'stop_b_step': 0.1,
    'target_a_start': 3.2,
    'target_a_end': 3.2,
    'target_a_step': 0.1,
    'target_b_start': 0,
    'target_b_end': 0,
    'target_b_step': 1.0,
}


def wrapped_compute_trades_daterange(start_date, end_date, parameter_configuration, bars_with_derivatives, use_trailing_stop=False, function_name='compute_trades_daterange()', use_cache=False, cache_directory=f'{OUTPUT_DIR}/trades.pickle'):
    if use_cache:
        print(f'Retrieved {function_name} from cache')
        return pandas.read_pickle(cache_directory)

    print(f'Starting {function_name}')
    start = time.time()
    ret = compute_trades_daterange(start_date, end_date, parameter_configuration, bars_with_derivatives, max_workers=24, use_trailing_stop=use_trailing_stop)
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
    ret = compute_all_parameter_stats(all_trades, parameter_configuration, max_workers=24)
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
    # Compute bars with derivatives
    bars_with_derivatives = wrapped_compute_derivatives(BARS_FILE_PATH, use_cache=True)

    # Get trades
    trades = wrapped_compute_trades_daterange(START_DATE, END_DATE, PARAMETER_CONFIGURATION, bars_with_derivatives, use_trailing_stop=True, use_cache=False)

    # Add derivatives to trades
    trades = wrapped_populate_derivatives(trades, bars_with_derivatives, use_cache=False)


    # Compute stats for short trades
    short_trades = trades[trades['direction'] == 'short']
    short_stats = wrapped_compute_all_parameter_stats(short_trades, PARAMETER_CONFIGURATION, function_name='compute_all_parameter_stats(shorts)', cache_directory=f'{OUTPUT_DIR}/stats_shorts.pickle')

    # Compute stats for long trades
    long_trades = trades[trades['direction'] == 'long']
    long_stats = wrapped_compute_all_parameter_stats(long_trades, PARAMETER_CONFIGURATION, function_name='compute_all_parameter_stats(longs)', cache_directory=f'{OUTPUT_DIR}/stats_longs.pickle')
