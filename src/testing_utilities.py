import pandas

def generate_quotes(points, step=0.01):
    output = []
    for x in range(len(points) - 1):
        start_point = points[x]
        end_point = points[x + 1]

        up = end_point > start_point
        current_point = start_point
        while current_point <= end_point if up else current_point >= end_point:
            current_point = round(float(current_point), 2)
            if not output or current_point != output[-1]:
                output.append(current_point)
            current_point += step if up else -1 * step

    return output


def generate_timestamps(count, start_timestamp='2020-12-10 14:30:00.000000+0000', second_step=1):
    current_timestamp = pandas.Timestamp(start_timestamp)

    output = []
    while len(output) < count:
        output.append(current_timestamp)
        current_timestamp = current_timestamp + pandas.Timedelta(second_step, 'seconds')

    return output


def generate_quotes_dataframe(points, quotes_step=0.01, timestamp_seconds_step=1, quotes_spread=0.01):
    ask_quotes = generate_quotes(points, quotes_step)
    bid_quotes = [round(ask - quotes_spread, 2) for ask in ask_quotes]
    timestamps = generate_timestamps(len(ask_quotes), second_step=timestamp_seconds_step)
    test_quotes = \
        {
            'timestamp': timestamps,
            'ask_price': ask_quotes,
            'bid_price': bid_quotes
        }

    df = pandas.DataFrame(data=test_quotes)
    df.set_index('timestamp', inplace=True)

    return df
