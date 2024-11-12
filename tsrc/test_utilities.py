from src.testing_utilities import *
from src.utilities import compute_trades_internal


# Test target hit on short 1
def test_target_hit_on_short_1(snapshot):
    buffer = 0
    stop = 1
    target = 1
    levels = [2, 0.9]
    points = [1, 2, 1, 3]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test target hit on short 2
def test_target_hit_on_short_2(snapshot):
    buffer = 0.1
    stop = 1
    target = 1
    levels = [2, 0.9]
    points = [1, 2, 0.7, 3]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test stop hit on short 1
def test_stop_hit_on_short_1(snapshot):
    buffer = 0
    stop = 1
    target = 1
    levels = [2, 0.9]
    points = [1, 2, 1.1, 3]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test stop hit on short 2
def test_stop_hit_on_short_2(snapshot):
    buffer = 0
    stop = 1
    target = 1
    levels = [2, 0.9]
    points = [1, 2, 3.5]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test time close on short 1
def test_time_close_on_short_1(snapshot):
    buffer = 0
    stop = 1
    target = 1.1
    levels = [2, 0.9]
    points = [1, 2, 1, 1.5, 1, 1.5, 1, 1.5]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test time close on short 2
def test_time_close_on_short_2(snapshot):
    buffer = 0
    stop = 1
    target = 1.1
    levels = [2, 0.9]
    points = [1, 2, 1, 1.5, 1, 1.5, 1, 2.9]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test target hit on long 1
def test_target_hit_on_long_1(snapshot):
    buffer = 0
    stop = 1
    target = 1
    levels = [2, 0.9]
    points = [3.01, 2.01, 3.01, 1.01]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test target hit on long 2
def test_target_hit_on_long_2(snapshot):
    buffer = 0.1
    stop = 1
    target = 1
    levels = [2, 0.9]
    points = [3.01, 2.01, 3.31, 2.91]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test stop hit on long 1
def test_stop_hit_on_long_1(snapshot):
    buffer = 0
    stop = 1
    target = 1
    levels = [2, 0.9]
    points = [3.01, 2.01, 2.91, 1.01]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test stop hit on long 2
def test_stop_hit_on_long_2(snapshot):
    buffer = 0
    stop = 1
    target = 1
    levels = [2, 0.1]
    points = [3.01, 2.01, 0.51]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test time close on long 1
def test_time_close_on_long_1(snapshot):
    buffer = 0
    stop = 1
    target = 1.1
    levels = [2, 0.9]
    points = [3.01, 2.01, 3.01, 2.51, 3.01, 2.51, 3.01, 2.51]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test time close on long 2
def test_time_close_on_long_2(snapshot):
    buffer = 0
    stop = 1
    target = 1.1
    levels = [2, 0.9]
    points = [3.01, 2.01, 3.01, 2.51, 3.01, 2.51, 3.01, 1.11]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test long and short on same day
def test_long_and_short_on_same_day(snapshot):
    buffer = 0
    stop = 2
    target = 1
    levels = [5, 15]
    points = [10, 4, 16, 10]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test 2 longs on same day
def test_2_longs_on_same_day(snapshot):
    buffer = 0
    stop = 2
    target = 2
    levels = [4, 2]
    points = [10, 4, 1, 20]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test multi parameter (short) 1
def test_multi_parameter_short_1(snapshot):
    levels = [10]
    points = [5, 12, 5]

    parameter_configuration = {
        'stop_start': 1,
        'stop_end': 4,
        'stop_step': 1,
        'buffer_start': 0,
        'buffer_end': 1,
        'buffer_step': 1,
        'target_start': 1,
        'target_end': 10,
        'target_step': 9,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test multi parameter (long) 2
def test_multi_parameter_long_2(snapshot):
    levels = [10]
    points = [15.01, 8.01, 15.01]

    parameter_configuration = {
        'stop_start': 1,
        'stop_end': 4,
        'stop_step': 1,
        'buffer_start': 0,
        'buffer_end': 1,
        'buffer_step': 1,
        'target_start': 1,
        'target_end': 10,
        'target_step': 9,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test long and short skip level
def test_long_and_short_skip_level(snapshot):
    buffer = 0
    stop = 1
    target = 1
    levels = [5, 15]
    points = [10, 3, 17, 10]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points, quotes_spread=0.0)

    # AUGMENT QUOTES FOR TESTING PURPOSES
    quotes = quotes[quotes['ask_price'] != 15]
    quotes = quotes[quotes['ask_price'] != 16.01]
    quotes = quotes[quotes['ask_price'] != 5]
    quotes = quotes[quotes['ask_price'] != 3.99]

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)


# Test levels on open quote
def test_levels_on_open_quote(snapshot):
    buffer = 0
    stop = 2
    target = 1
    levels = [10, 9.99]
    points = [10, 4, 16, 10]

    parameter_configuration = {
        'stop_start': stop,
        'stop_end': stop,
        'stop_step': 1,
        'buffer_start': buffer,
        'buffer_end': buffer,
        'buffer_step': 1,
        'target_start': target,
        'target_end': target,
        'target_step': 1,
    }

    quotes = generate_quotes_dataframe(points)

    actual = compute_trades_internal('01/01/1994', levels, quotes, parameter_configuration)

    snapshot.assert_match(actual)
