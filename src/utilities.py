import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import statistics
import pandas
import numpy as np
import pytz

from directory_definitions import *


def calc_dates(start_date, end_date):
    return [date.strftime("%Y-%m-%d") for date in pandas.date_range(start=start_date, end=end_date) if
            date.weekday() <= 4]


def load_levels():
    levels_df = pandas.read_csv(f'{DOWNLOADS_DIR}/levels/levels.csv')
    date_level_pairs = [(x, y) for x, y in zip(levels_df['Date'], levels_df['Level'])]

    d = {}
    for key, value in date_level_pairs:
        if key not in d:
            d[key] = []
        d[key].append(value)

    return d


def load_quotes(date):
    return pandas.read_pickle(f'{DOWNLOADS_DIR}/clean_quotes/SPY_{date}.pickle')


def split(unsplit_list, num_chunks):
    max_sublist_len = math.ceil(len(unsplit_list) / num_chunks)
    return [unsplit_list[i:i + max_sublist_len] for i in range(0, len(unsplit_list), max_sublist_len)]


def count_parameters(parameter_configuration):
    min_distance_range = [round(x, 2) for x in np.arange(parameter_configuration['min_distance_start'],
                                                         parameter_configuration['min_distance_end'] + (
                                                                     parameter_configuration['min_distance_step'] / 2),
                                                         parameter_configuration['min_distance_step'])]
    stop_range = [round(x, 2) for x in np.arange(parameter_configuration['stop_start'],
                                                 parameter_configuration['stop_end'] + (
                                                             parameter_configuration['stop_step'] / 2),
                                                 parameter_configuration['stop_step'])]
    buffer_range = [round(x, 2) for x in np.arange(parameter_configuration['buffer_start'],
                                                   parameter_configuration['buffer_end'] + (
                                                               parameter_configuration['buffer_step'] / 2),
                                                   parameter_configuration['buffer_step'])]
    target_range = [round(x, 2) for x in np.arange(parameter_configuration['target_start'],
                                                   parameter_configuration['target_end'] + (
                                                               parameter_configuration['target_step'] / 2),
                                                   parameter_configuration['target_step'])]

    return len(min_distance_range) * len(stop_range) * len(buffer_range) * len(target_range)


def compute_fuzzy_mean(series, days_lookback=5, days_lookback_buffer=5):
    ttl_lookback_distance = days_lookback + days_lookback_buffer

    samples = [volume for i in range(1, ttl_lookback_distance + 1) if not math.isnan((volume := series[f'dod_volume_{i}']))]

    # downselect to desired number of samples
    samples = samples[:days_lookback]

    count = len(samples)
    mean = statistics.fmean(samples) if count > 0 else float('nan')

    dod_volume_mean_column_name = f'dod_volume_mean_{days_lookback}'
    dod_volume_mean_count_column_name = f'dod_volume_mean_count_{days_lookback}'

    return {dod_volume_mean_column_name: mean, dod_volume_mean_count_column_name: count}


def calculate_active_hours(bars_df):
    start_hour = 9
    start_minute = 30
    end_hour = 16
    end_minute = 0

    bars_df['timestamp'] = bars_df.index

    bars_df['market_open'] = bars_df['timestamp'].map(
        lambda x: x.astimezone('US/Eastern').replace(hour=start_hour, minute=start_minute, second=0, microsecond=0, nanosecond=0).astimezone(pytz.utc))
    bars_df['market_close'] = bars_df['timestamp'].map(
        lambda x: x.astimezone('US/Eastern').replace(hour=end_hour, minute=end_minute, second=0, microsecond=0, nanosecond=0).astimezone(pytz.utc))

    bars_df['active_hours'] = (bars_df['timestamp'] >= bars_df['market_open']) & (
                bars_df['timestamp'] < bars_df['market_close'])

    bars_df.drop(['timestamp', 'market_open', 'market_close'], axis=1, inplace=True)


def calculate_atr(bars_df, tr_sma_minutes=20):
    cr = bars_df['high'] - bars_df['low']
    high_close = np.abs(bars_df['high'] - bars_df['close'].shift(1, pandas.Timedelta(1, 'm')))
    low_close = np.abs(bars_df['low'] - bars_df['close'].shift(1, pandas.Timedelta(1, 'm')))

    atr_options = pandas.concat([cr, high_close, low_close], axis=1)

    tr = np.max(atr_options, axis=1)

    bars_df['tr'] = tr

    bars_df[f'tr_SMA_{tr_sma_minutes}min'] = bars_df[bars_df['active_hours']]['tr'].rolling(f'{tr_sma_minutes}min', min_periods=1).mean()


def calculate_volume_sma(bars_df, volume_sma_minutes=20):
    bars_df[f'volume_SMA_{volume_sma_minutes}min'] = bars_df[bars_df['active_hours']]['volume'].rolling(f'{volume_sma_minutes}min', min_periods=1).mean()


def compute_derivatives(bars_file_path, days_lookback=5, days_lookback_buffer=10, rvol_sma_minutes=15, tr_sma_minutes=20, volume_sma_minutes=20):
    df = pandas.read_pickle(bars_file_path)

    # Don't perform inplace
    df = df.copy()

    ttl_lookback_distance = days_lookback + days_lookback_buffer
    for i in range(1, ttl_lookback_distance + 1):
        column_name = f'dod_volume_{i}'
        time_delta = pandas.to_timedelta(i, 'd')
        df[column_name] = df.apply(lambda x: df.loc[x.name - time_delta].volume if (x.name - time_delta) in df.index else float('nan'), axis=1)

    mean_count_df = df.apply(lambda x: compute_fuzzy_mean(x, days_lookback, days_lookback_buffer), result_type='expand', axis=1)

    df = pandas.concat([df, mean_count_df], axis='columns')

    # Compute and set rvol
    rvol_column_name = f'rvol_{days_lookback}'
    df[rvol_column_name] = df['volume'] / df[f'dod_volume_mean_{days_lookback}']

    # Compute and set rvol moving averages
    rvol_sma_column_name = f'{rvol_column_name}_SMA_{rvol_sma_minutes}min'
    df[rvol_sma_column_name] = df[rvol_column_name].rolling(f'{rvol_sma_minutes}min', min_periods=1).mean()

    # Calculate active hours
    calculate_active_hours(df)

    # Compute volume sma
    calculate_volume_sma(df, volume_sma_minutes)
    volume_sma_column_name = f'volume_SMA_{volume_sma_minutes}min'

    # Compute ATR
    calculate_atr(df, tr_sma_minutes)
    tr_sma_column_name = f'tr_SMA_{tr_sma_minutes}min'

    # Drop columns that are no longer needed
    dod_volume_mean_count_column_name = f'dod_volume_mean_count_{days_lookback}'
    df = df[['open', 'high', 'low', 'close', 'volume', 'trade_count', dod_volume_mean_count_column_name, rvol_column_name, rvol_sma_column_name, volume_sma_column_name, 'tr', tr_sma_column_name]]

    return df


def populate_derivatives(trades_df, derivative_complete_bars_df, rvol_days_lookback=5, rvol_sma_minutes=15, tr_sma_minutes=20, volume_sma_minutes=20):
    trades_df = trades_df.copy()

    # All columns we want to map from derivative_complete_bars_df
    dod_volume_mean_count_column_name = f'dod_volume_mean_count_{rvol_days_lookback}'
    rvol_column_name = f'rvol_{rvol_days_lookback}'
    rvol_sma_column_name = f'rvol_{rvol_days_lookback}_SMA_{rvol_sma_minutes}min'
    volume_sma_column_name = f'volume_SMA_{volume_sma_minutes}min'
    tr_sma_column_name = f'tr_SMA_{tr_sma_minutes}min'

    # Downselect only these columns
    derivative_complete_bars_df = derivative_complete_bars_df[
        [dod_volume_mean_count_column_name, rvol_column_name, rvol_sma_column_name, volume_sma_column_name, 'tr', tr_sma_column_name]]

    # Compute entry_time_minute
    trades_df['entry_time_previous_minute'] = trades_df.apply(
        lambda x: x['entry_time'].replace(second=0, microsecond=0, nanosecond=0) - pandas.Timedelta(minutes=1), axis=1)

    # Merge in desired columns
    trades_with_derivatives = pandas.merge(trades_df, derivative_complete_bars_df, how='left', left_on='entry_time_previous_minute', right_index=True)

    # Drop columns that are no longer needed
    trades_with_derivatives.drop(columns=['entry_time_previous_minute'], inplace=True)

    return trades_with_derivatives


def calc_max_drawdown(trades):
    total_return_r = trades['r']
    total_return_r = pandas.concat([pandas.Series([0]), total_return_r])

    cumsum = total_return_r.cumsum()
    cumsum_max = cumsum.cummax()
    drawdown = cumsum_max - cumsum

    max_drawdown = drawdown.max()

    return max_drawdown


def compute_stats(parameter_chunk, df, tr_sma_minutes=20):
    stats = []
    for parameter in parameter_chunk:
        # (min_distance, max_rvol, buffer, stop, target)
        min_distance = parameter[0]
        max_rvol = parameter[1]
        buffer = parameter[2]
        stop_parameters = parameter[3]
        target_parameters = parameter[4]

        stop_a = stop_parameters[0]
        stop_b = stop_parameters[1]

        target_a = target_parameters[0]
        target_b = target_parameters[1]

        # Select relevant trades
        _df = df[(df['distance'] >= min_distance) &
                 (df['rvol_5_SMA_15min'] <= max_rvol) &
                 (df['parameter_buffer'] == buffer) &
                 (df['parameter_stop_a'] == stop_a) &
                 (df['parameter_stop_b'] == stop_b) &
                 (df['parameter_target_a'] == target_a) &
                 (df['parameter_target_b'] == target_b)]

        # Hit count
        hit_count = _df[~_df['r'].isnull()].shape[0]

        if hit_count <= 0:
            continue

        # Win rate
        win_rate = (_df[_df['r'] >= 0].shape[0] / hit_count) if hit_count > 0 else -1
        win_percent = win_rate * 100

        # Time close rate
        time_close_rate = (_df[_df['time_close']].shape[0] / hit_count) if hit_count > 0 else -1
        time_close_percent = time_close_rate * 100

        # Total return
        total_return_r = _df['r'].sum()

        # Median Stop, Target
        median_stop = _df['stop'].median()
        median_target = _df['target'].median()

        # Median ATR
        tr_sma_column_name = f'tr_SMA_{tr_sma_minutes}min'
        median_tr_sma = _df[tr_sma_column_name].median()

        # Get set of all winners and all losers
        winning_trades_r = _df[_df['r'] > 0]['r']
        losing_trades_r = _df[_df['r'] < 0]['r']

        # Profit factor
        profit_r_sum = winning_trades_r.sum()
        loss_r_sum = abs(losing_trades_r.sum())
        profit_factor = profit_r_sum / loss_r_sum if loss_r_sum != 0 else -1

        # Real r mean
        profit_r_mean = winning_trades_r.mean()
        loss_r_mean = abs(losing_trades_r.mean())
        real_r_mean = profit_r_mean / loss_r_mean

        # Real r median
        profit_r_median = winning_trades_r.median()
        loss_r_median = abs(losing_trades_r.median())
        real_r_median = profit_r_median / loss_r_median

        # Kelly value
        kelly = win_rate - ((1 - win_rate) / real_r_mean)

        # Kelly adjusted return
        kelly_adj_return = kelly * total_return_r

        # Max drawdown
        max_drawdown = calc_max_drawdown(_df)

        # Return to drawdown ratio
        return_to_dd_ratio = total_return_r / max_drawdown if max_drawdown > 0 else float('nan')

        stat = {'min_distance': min_distance,
                'max_rvol': max_rvol,
                'buffer': buffer,
                'stop_a': stop_a,
                'stop_b': stop_b,
                'target_a': target_a,
                'target_b': target_b,
                'median_stop': round(median_stop, 2),
                'median_target': round(median_target, 2),
                'median_tr_sma': round(median_tr_sma, 2),
                'real_r_mean': round(real_r_mean, 2),
                'real_r_median': round(real_r_median, 2),
                'loss_r_mean': round(loss_r_mean, 2),
                'loss_r_median': round(loss_r_median, 2),
                'time_close_rate': round(time_close_percent, 2),
                'hit_count': hit_count,
                'win_rate': round(win_percent, 2),
                'total_return_r': round(total_return_r, 2),
                'total_positive_return_r': round(profit_r_sum, 2),
                'total_loss_return_r': round(loss_r_sum, 2),
                'profit_factor': round(profit_factor, 2),
                'kelly': round(kelly, 2),
                'kelly_adj_return': round(kelly_adj_return, 2),
                'max_drawdown': round(max_drawdown, 2),
                'return_to_dd_ratio': round(return_to_dd_ratio, 2),
                }
        stats.append(stat)

    return stats


def compute_all_parameter_stats(raw_trades, parameter_configuration, max_workers=24):
    df = raw_trades

    # Generate parameters
    min_distance_range = generate_range(parameter_configuration['min_distance_start'], parameter_configuration['min_distance_end'], parameter_configuration['min_distance_step'])
    max_rvol_range = generate_range(parameter_configuration['max_rvol_start'], parameter_configuration['max_rvol_end'], parameter_configuration['max_rvol_step'])

    buffer_range = generate_range(parameter_configuration['buffer_start'], parameter_configuration['buffer_end'], parameter_configuration['buffer_step'])
    stop_a_range = generate_range(parameter_configuration['stop_a_start'], parameter_configuration['stop_a_end'], parameter_configuration['stop_a_step'])
    stop_b_range = generate_range(parameter_configuration['stop_b_start'], parameter_configuration['stop_b_end'], parameter_configuration['stop_b_step'])
    target_a_range = generate_range(parameter_configuration['target_a_start'], parameter_configuration['target_a_end'], parameter_configuration['target_a_step'])
    target_b_range = generate_range(parameter_configuration['target_b_start'], parameter_configuration['target_b_end'], parameter_configuration['target_b_step'])

    stop_parameters_range = generate_combinations(stop_a_range, stop_b_range)
    target_parameters_range = generate_combinations(target_a_range, target_b_range)

    # TMP!
    stop_parameters_range = downselect_combinations(stop_parameters_range, 0.3, 0.2, 1.0)
    target_parameters_range = downselect_combinations(target_parameters_range, 0.3, 0.25, 4.0)


    # Gather all parameter combinations as tuples (min_distance, max_rvol, buffer, stop, target)
    # Split into chunks, one for each processor
    parameters = []
    for min_distance in min_distance_range:
        for max_rvol in max_rvol_range:
            for buffer in buffer_range:
                for stop_parameters in stop_parameters_range:
                    for target_parameters in target_parameters_range:
                        parameters.append((min_distance, max_rvol, buffer, stop_parameters, target_parameters))
    parameters_chunks = split(parameters, max_workers)

    # Iterate all parameter chunks and assign to a processor core
    stats = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for parameters_chunk in parameters_chunks:
            # future = executor.submit(compute_stats, parameters_chunk, df.copy())
            future = executor.submit(compute_stats, parameters_chunk, df)
            futures.append(future)

        count = 0
        for future in as_completed(futures):
            count = count + 1
            print(f'{count} / {len(futures)} completed')
            try:
                stats_chunk = future.result()
                stats.extend(stats_chunk)
            except Exception as ex:
                print('Exception occurred', flush=True)
                print(ex, flush=True)
                traceback.print_exc()

    stats_df = pandas.DataFrame(stats)
    stats_df.sort_values(by=['min_distance', 'max_rvol', 'buffer', 'stop_a', 'stop_b', 'target_a', 'target_b'], inplace=True)
    stats_df.reset_index(drop=True, inplace=True)

    return stats_df


def compute_stats_distance_bucketed(parameter_chunk, df):
    stats = []
    for parameter in parameter_chunk:
        # (distance_bucket_pair, max_rvol, buffer, stop, target)
        distance_bucket_pair = parameter[0]
        distance_min_inclusive = distance_bucket_pair[0]
        distance_max_exclusive = distance_bucket_pair[1]
        max_rvol = parameter[1]
        buffer = parameter[2]
        stop = parameter[3]
        target = parameter[4]

        # Select relevant trades
        _df = df[(df['distance'] >= distance_min_inclusive) & (df['distance'] < distance_max_exclusive) &
                 (df['rvol_5_SMA_15min'] <= max_rvol) & (df['parameter_buffer'] == buffer) &
                 (df['parameter_stop'] == stop) & (df['parameter_target'] == target)]

        # Hit count
        hit_count = _df[~_df['r'].isnull()].shape[0]

        # Win rate
        win_rate = (_df[_df['r'] >= 0].shape[0] / hit_count) if hit_count > 0 else -1
        win_percent = win_rate * 100

        # Time close rate
        time_close_rate = (_df[_df['time_close']].shape[0] / hit_count) if hit_count > 0 else -1
        time_close_percent = time_close_rate * 100

        # Total return
        total_return_r = _df['r'].sum()


        # Get set of all winners and all losers
        winning_trades = _df[_df['r'] > 0]['r']
        losing_trades = _df[_df['r'] < 0]['r']

        # Profit factor
        profit_sum = winning_trades.sum()
        loss_sum = abs(losing_trades.sum())
        profit_factor = profit_sum / loss_sum if loss_sum != 0 else -1

        # Target r
        target_r = target / stop

        # Real r mean
        profit_mean = winning_trades.mean()
        loss_mean = abs(losing_trades.mean())
        real_r_mean = profit_mean / loss_mean

        # Real r median
        profit_median = winning_trades.median()
        loss_median = abs(losing_trades.median())
        real_r_median = profit_median / loss_median

        # Kelly value
        kelly = win_rate - ((1 - win_rate) / real_r_mean)

        # Kelly adjusted return
        kelly_adj_return = kelly * total_return_r

        stat = {'distance_key': f'{distance_min_inclusive} -> {distance_max_exclusive}',
                'distance_min': distance_min_inclusive,
                'distance_max': distance_max_exclusive,
                'max_rvol': max_rvol,
                'buffer': buffer,
                'stop': stop,
                'target': target,
                'target_r': round(target_r, 3),
                'real_r_mean': round(real_r_mean, 3),
                'real_r_median': round(real_r_median, 3),
                'time_close_rate': round(time_close_percent, 3),
                'hit_count': hit_count,
                'win_rate': round(win_percent, 3),
                'total_return_r': round(total_return_r, 3),
                'profit_factor': round(profit_factor, 3),
                'kelly': round(kelly, 3),
                'kelly_adj_return': round(kelly_adj_return, 3)}
        stats.append(stat)

    return stats


def compute_all_parameter_stats_distance_bucketed(raw_trades, parameter_configuration, max_workers=24):
    df = raw_trades

    # Generate parameters
    distance_bucket_points = parameter_configuration['distance_bucket_points']
    distance_bucket_pairs = [*zip(distance_bucket_points, distance_bucket_points[1:])]

    stop_range = [round(x, 2) for x in np.arange(parameter_configuration['stop_start'],
                                                 parameter_configuration['stop_end'] + (
                                                             parameter_configuration['stop_step'] / 2),
                                                 parameter_configuration['stop_step'])]
    max_rvol_range = [round(x, 2) for x in np.arange(parameter_configuration['max_rvol_start'],
                                                         parameter_configuration['max_rvol_end'] + (
                                                                     parameter_configuration['max_rvol_step'] / 2),
                                                         parameter_configuration['max_rvol_step'])]
    buffer_range = [round(x, 2) for x in np.arange(parameter_configuration['buffer_start'],
                                                   parameter_configuration['buffer_end'] + (
                                                               parameter_configuration['buffer_step'] / 2),
                                                   parameter_configuration['buffer_step'])]
    target_range = [round(x, 2) for x in np.arange(parameter_configuration['target_start'],
                                                   parameter_configuration['target_end'] + (
                                                               parameter_configuration['target_step'] / 2),
                                                   parameter_configuration['target_step'])]

    # Gather all parameter combinations as tuples (distance_bucket_pair, max_rvol, buffer, stop, target)
    # Split into chunks, one for each processor
    parameters = []
    for distance_bucket_pair in distance_bucket_pairs:
        for max_rvol in max_rvol_range:
            for buffer in buffer_range:
                for stop in stop_range:
                    for target in target_range:
                        parameters.append((distance_bucket_pair, max_rvol, buffer, stop, target))
    parameters_chunks = split(parameters, max_workers)

    # Iterate all parameter chunks and assign to a processor core
    stats = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for parameters_chunk in parameters_chunks:
            future = executor.submit(compute_stats_distance_bucketed, parameters_chunk, df.copy())
            futures.append(future)

        for future in futures:
            try:
                stats_chunk = future.result()
                stats.extend(stats_chunk)
            except Exception as ex:
                print('Exception occurred', flush=True)
                print(ex, flush=True)
                traceback.print_exc()

    return pandas.DataFrame(stats)


def generate_range(start, end, step, round_decimal=2):
    return [round(x, round_decimal) for x in np.arange(start, end + (step / 2), step)]


def generate_combinations(a_range, b_range):
    return [(a, b) for a in a_range for b in b_range]


def downselect_combinations(combinations, median_x, min_y, max_y):
    new_combinations = [combination for combination in combinations if (y := (combination[0] + (median_x * combination[1]))) >= min_y and y <= max_y]

    # print(f'Reduced from {len(combinations)} to {len(new_combinations)}')
    # print(combinations)
    # print(new_combinations)

    return new_combinations


def compute_trades_daterange(start_date, end_date, parameter_configuration, bars_with_derivatives, max_workers=24, use_trailing_stop=False):
    dates = calc_dates(start_date, end_date)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for date in dates:
            future = executor.submit(compute_trades, date, parameter_configuration, bars_with_derivatives.copy(), use_trailing_stop)
            futures.append(future)

        count = 0
        for future in as_completed(futures):
            count = count + 1
            print(f'{count} / {len(futures)} completed')
            try:
                rr = future.result()
                results.extend(rr)
            except Exception as ex:
                print('Exception occurred', flush=True)
                print(ex, flush=True)
                traceback.print_exc()

    # Create trades dataframe
    trades = pandas.DataFrame(results)

    # Sort trades
    trades.sort_values(by='entry_time', ascending=True, inplace=True)

    return trades


def compute_trades(date, parameter_configuration, bars_with_derivatives, use_trailing_stop):
    # Get levels
    levels = load_levels().get(date, [])

    # Get quotes
    quotes = load_quotes(date)

    return compute_trades_internal(date, levels, quotes, parameter_configuration, bars_with_derivatives, use_trailing_stop)

def compute_trades_internal(date, levels, quotes, parameter_configuration, bars_with_derivatives, use_trailing_stop, tr_sma_minutes=20, default_tr_sma=0.35):
    # If no levels or data, exit
    if not levels or quotes.empty:
        return []

    # Populate constants
    bars_tr_sma_column_name = f'tr_SMA_{tr_sma_minutes}min'

    # Massage quotes
    quotes = quotes.sort_index(ascending=True)
    quotes.reset_index(inplace=True)

    # Get starting quote data
    start_ask = quotes.iloc[0]['ask_price']
    start_bid = quotes.iloc[0]['bid_price']
    start_mid = (start_ask + start_bid) / 2

    # Generate parameters
    buffer_range = generate_range(parameter_configuration['buffer_start'], parameter_configuration['buffer_end'], parameter_configuration['buffer_step'])
    stop_a_range = generate_range(parameter_configuration['stop_a_start'], parameter_configuration['stop_a_end'], parameter_configuration['stop_a_step'])
    stop_b_range = generate_range(parameter_configuration['stop_b_start'], parameter_configuration['stop_b_end'], parameter_configuration['stop_b_step'])
    target_a_range = generate_range(parameter_configuration['target_a_start'], parameter_configuration['target_a_end'], parameter_configuration['target_a_step'])
    target_b_range = generate_range(parameter_configuration['target_b_start'], parameter_configuration['target_b_end'], parameter_configuration['target_b_step'])

    stop_parameters_range = generate_combinations(stop_a_range, stop_b_range)
    target_parameters_range = generate_combinations(target_a_range, target_b_range)

    # TMP!
    stop_parameters_range = downselect_combinations(stop_parameters_range, 0.3, 0.2, 1.0)
    target_parameters_range = downselect_combinations(target_parameters_range, 0.3, 0.25, 4.0)


    completed_orders = []
    for buffer in buffer_range:
        short_orders = []
        long_orders = []

        # Generate orders
        for level in levels:
            short_entry = level - buffer
            long_entry = level + buffer

            if short_entry > max(start_mid, start_ask):
                short_orders.append(short_entry)
            elif long_entry < min(start_mid, start_bid):
                long_orders.append(long_entry)

        # Continue to next buffer if there's no orders to be taken
        if not (short_orders or long_orders):
            continue

        # Iterate short_orders and execute trades
        for short_order in short_orders:
            # Find entries
            entries = quotes[quotes['ask_price'] >= short_order]

            # No entry, move to next order
            if entries.empty:
                continue

            # Derive trade values
            entry_series = entries.iloc[0]
            trade_entry_time = entry_series['timestamp']
            trade_entry_price = entry_series['ask_price']

            # Get tr_sma from bars
            trade_entry_time_previous_minute = trade_entry_time.replace(second=0, microsecond=0, nanosecond=0) - pandas.Timedelta(minutes=1)
            tr_sma = bars_with_derivatives.loc[trade_entry_time_previous_minute][bars_tr_sma_column_name]

            default_tr_sma_used = False
            if math.isnan(tr_sma):
                tr_sma = default_tr_sma
                default_tr_sma_used = True

            # Get quotes after (and including) entry
            quotes_entry_down_select = quotes.loc[entry_series.name:]

            # Iterate and evaluate stops
            for stop_parameters in stop_parameters_range:
                # Calculate stop
                stop_a = stop_parameters[0]
                stop_b = stop_parameters[1]
                stop = round(stop_a + (tr_sma * stop_b), 2)

                if stop == 0:
                    continue

                trade_stop_price = trade_entry_price + stop

                # Get dataframe before (and including) stop
                if not use_trailing_stop:
                    exits = quotes_entry_down_select[quotes_entry_down_select['ask_price'] >= trade_stop_price]
                else:
                    FRACTION = 4

                    quotes_entry_down_select = quotes_entry_down_select.iloc[:]

                    quotes_entry_down_select['distance'] = trade_entry_price - quotes_entry_down_select['ask_price']
                    quotes_entry_down_select['max_distance'] = quotes_entry_down_select['distance'].cummax()

                    terminal_stop = round(stop / FRACTION, 2)
                    initial_stop_price = trade_entry_price + stop

                    quotes_entry_down_select['stop_price'] = initial_stop_price

                    move_stop_triggered = quotes_entry_down_select[quotes_entry_down_select['max_distance'] >= stop]
                    if not move_stop_triggered.empty:
                        move_stop_triggered_series = move_stop_triggered.iloc[0]
                        before_trigger_df = quotes_entry_down_select.loc[:move_stop_triggered_series.name]

                        max_price = before_trigger_df['ask_price'].max()
                        new_stop_price = round(max_price + terminal_stop, 2)

                        if new_stop_price < initial_stop_price:
                            quotes_entry_down_select.loc[move_stop_triggered_series.name:, 'stop_price'] = new_stop_price

                    exits = quotes_entry_down_select[quotes_entry_down_select['ask_price'] >= quotes_entry_down_select['stop_price']]


                # Downselect to (and including) stop, otherwise no stop exists
                if not exits.empty:
                    exit_series = exits.iloc[0]
                    quotes_exit_down_select = quotes_entry_down_select.loc[:exit_series.name]
                    trade_stop_in_play = True
                else:
                    exit_series = quotes_entry_down_select.iloc[-1]
                    quotes_exit_down_select = quotes_entry_down_select
                    trade_stop_in_play = False

                # Calculate max target
                max_target_series = quotes_exit_down_select.loc[quotes_exit_down_select['ask_price'].idxmin()]
                trade_max_target_exit_time = max_target_series['timestamp']
                trade_max_target_exit_price = max_target_series['ask_price']

                # Iterate and evaluate targets, generate trades
                for target_parameters in target_parameters_range:
                    # Calculate target
                    target_a = target_parameters[0]
                    target_b = target_parameters[1]
                    target = round(target_a + (tr_sma * target_b), 2)

                    if target == 0:
                        continue

                    trade_target_price = trade_entry_price - target

                    # Evaluate trade pass/fail
                    # - Target is hit
                    if trade_target_price >= trade_max_target_exit_price:
                        trade_stop_hit = False
                        trade_target_hit = True
                        trade_time_close = False
                        trade_exit_price = trade_target_price
                    # - Stop is hit
                    elif trade_stop_in_play:
                        trade_stop_hit = True
                        trade_target_hit = False
                        trade_time_close = False
                        trade_exit_price = exit_series['ask_price']
                    # - Trade close at EOD (or end of data provided)
                    else:
                        trade_stop_hit = False
                        trade_target_hit = False
                        trade_time_close = True
                        trade_exit_price = exit_series['ask_price']

                    # Create trade and save
                    risk = trade_stop_price - trade_entry_price

                    completed_order = \
                        {
                            'parameter_buffer': buffer,
                            'parameter_stop_a': stop_a,
                            'parameter_stop_b': stop_b,
                            'parameter_target_a': target_a,
                            'parameter_target_b': target_b,
                            'tr_SMA': round(tr_sma, 2),
                            'default_tr_sma_used': default_tr_sma_used,
                            'stop': stop,
                            'target': target,
                            'target_r': round(target / stop, 2),
                            'distance': round(short_order - start_mid, 2),
                            'direction': 'short',
                            'date': date,
                            'entry_price': round(trade_entry_price, 2),
                            'entry_order_price': round(short_order, 2),
                            'exit_price': round(trade_exit_price, 2),
                            'stop_hit': trade_stop_hit,
                            'target_hit': trade_target_hit,
                            'time_close': trade_time_close,
                            'target_price': round(trade_target_price, 2),
                            'stop_price': round(trade_stop_price, 2),
                            'max_target_exit_price': round(trade_max_target_exit_price, 2),
                            'entry_time': trade_entry_time,
                            'max_target_exit_time': trade_max_target_exit_time,
                            'max_distance': round(trade_entry_price - trade_max_target_exit_price, 2),
                            'max_r': round((trade_entry_price - trade_max_target_exit_price) / risk, 2),
                            'theoretical_r': round((short_order - trade_exit_price) / risk, 2),
                            'r': round((trade_entry_price - trade_exit_price) / risk, 2),
                        }
                    completed_orders.append(completed_order)

        # Iterate long_orders and execute trades
        for long_order in long_orders:
            # Find entries
            entries = quotes[quotes['bid_price'] <= long_order]

            # No entry, move to next order
            if entries.empty:
                continue

            # Derive trade values
            entry_series = entries.iloc[0]
            trade_entry_time = entry_series['timestamp']
            trade_entry_price = entry_series['bid_price']

            # Get tr_sma from bars
            trade_entry_time_previous_minute = trade_entry_time.replace(second=0, microsecond=0, nanosecond=0) - pandas.Timedelta(minutes=1)
            tr_sma = bars_with_derivatives.loc[trade_entry_time_previous_minute][bars_tr_sma_column_name]

            default_tr_sma_used = False
            if math.isnan(tr_sma):
                tr_sma = default_tr_sma
                default_tr_sma_used = True

            # Get quotes after (and including) entry
            quotes_entry_down_select = quotes.loc[entry_series.name:]

            # Iterate and evaluate stops
            for stop_parameters in stop_parameters_range:
                # Calculate stop
                stop_a = stop_parameters[0]
                stop_b = stop_parameters[1]
                stop = round(stop_a + (tr_sma * stop_b), 2)

                if stop == 0:
                    continue

                trade_stop_price = trade_entry_price - stop

                # Get dataframe before (and including) stop
                if not use_trailing_stop:
                    exits = quotes_entry_down_select[quotes_entry_down_select['bid_price'] <= trade_stop_price]
                else:
                    FRACTION = 4

                    quotes_entry_down_select = quotes_entry_down_select.iloc[:]

                    quotes_entry_down_select['distance'] = quotes_entry_down_select['bid_price'] - trade_entry_price
                    quotes_entry_down_select['max_distance'] = quotes_entry_down_select['distance'].cummax()

                    terminal_stop = round(stop / FRACTION, 2)
                    initial_stop_price = trade_entry_price - stop

                    quotes_entry_down_select['stop_price'] = initial_stop_price

                    move_stop_triggered = quotes_entry_down_select[quotes_entry_down_select['max_distance'] >= stop]
                    if not move_stop_triggered.empty:
                        move_stop_triggered_series = move_stop_triggered.iloc[0]
                        before_trigger_df = quotes_entry_down_select.loc[:move_stop_triggered_series.name]

                        min_price = before_trigger_df['bid_price'].min()
                        new_stop_price = round(min_price - terminal_stop, 2)

                        if new_stop_price > initial_stop_price:
                            quotes_entry_down_select.loc[move_stop_triggered_series.name:, 'stop_price'] = new_stop_price
                            # print(f'Moving stop {quotes_entry_down_select["stop_price"].unique()}')
                        # else:
                            # print(f'? stop {initial_stop_price} -> {new_stop_price}')
                    exits = quotes_entry_down_select[quotes_entry_down_select['bid_price'] <= quotes_entry_down_select['stop_price']]


                # Downselect to (and including) stop, otherwise no stop exists
                if not exits.empty:
                    exit_series = exits.iloc[0]
                    quotes_exit_down_select = quotes_entry_down_select.loc[:exit_series.name]
                    trade_stop_in_play = True
                else:
                    exit_series = quotes_entry_down_select.iloc[-1]
                    quotes_exit_down_select = quotes_entry_down_select
                    trade_stop_in_play = False

                # Calculate max target
                max_target_series = quotes_exit_down_select.loc[quotes_exit_down_select['bid_price'].idxmax()]
                trade_max_target_exit_time = max_target_series['timestamp']
                trade_max_target_exit_price = max_target_series['bid_price']

                # Iterate and evaluate targets, generate trades
                for target_parameters in target_parameters_range:
                    # Calculate target
                    target_a = target_parameters[0]
                    target_b = target_parameters[1]
                    target = round(target_a + (tr_sma * target_b), 2)

                    if target == 0:
                        continue

                    trade_target_price = trade_entry_price + target

                    # Evaluate trade pass/fail
                    # - Target is hit
                    if trade_target_price <= trade_max_target_exit_price:
                        trade_stop_hit = False
                        trade_target_hit = True
                        trade_time_close = False
                        trade_exit_price = trade_target_price
                    # - Stop is hit
                    elif trade_stop_in_play:
                        trade_stop_hit = True
                        trade_target_hit = False
                        trade_time_close = False
                        trade_exit_price = exit_series['bid_price']
                    # - Trade close at EOD (or end of data provided)
                    else:
                        trade_stop_hit = False
                        trade_target_hit = False
                        trade_time_close = True
                        trade_exit_price = exit_series['bid_price']

                    # Create trade and save
                    risk = trade_entry_price - trade_stop_price

                    completed_order = \
                        {
                            'parameter_buffer': buffer,
                            'parameter_stop_a': stop_a,
                            'parameter_stop_b': stop_b,
                            'parameter_target_a': target_a,
                            'parameter_target_b': target_b,
                            'tr_SMA': round(tr_sma, 2),
                            'default_tr_sma_used': default_tr_sma_used,
                            'stop': stop,
                            'target': target,
                            'target_r': round(target / stop, 2),
                            'distance': round(start_mid - long_order, 2),
                            'direction': 'long',
                            'date': date,
                            'entry_price': round(trade_entry_price, 2),
                            'entry_order_price': round(long_order, 2),
                            'exit_price': round(trade_exit_price, 2),
                            'stop_hit': trade_stop_hit,
                            'target_hit': trade_target_hit,
                            'time_close': trade_time_close,
                            'target_price': round(trade_target_price, 2),
                            'stop_price': round(trade_stop_price, 2),
                            'max_target_exit_price': round(trade_max_target_exit_price, 2),
                            'entry_time': trade_entry_time,
                            'max_target_exit_time': trade_max_target_exit_time,
                            'max_distance': round(trade_max_target_exit_price - trade_entry_price, 2),
                            'max_r': round((trade_max_target_exit_price - trade_entry_price) / risk, 2),
                            'theoretical_r': round((trade_exit_price - long_order) / risk, 2),
                            'r': round((trade_exit_price - trade_entry_price) / risk, 2),
                        }
                    completed_orders.append(completed_order)

    return completed_orders
