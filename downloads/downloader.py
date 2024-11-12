import os.path
import pytz
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas
import alpaca_trade_api as api


START_DATE = '2023-02-02'
END_DATE = '2023-02-16'

QUOTES_FOLDER = 'full_day_quotes'

API_KEY = '<INSERT_API_KEY_HERE>'
SECRET_KEY = '<INSERT_SECRET_KEY_HERE>'


def generate_dates(start_date, end_date, start_hour=9, start_minute=30, end_hour=16, end_minute=0):
    dates_list = []
    for d in pandas.date_range(start=start_date, end=end_date):
        if d.weekday() > 4:
            continue
        start = pytz.timezone('US/Eastern').localize(d.replace(hour=start_hour, minute=start_minute)).astimezone(pytz.utc).isoformat()
        end = pytz.timezone('US/Eastern').localize(d.replace(hour=end_hour, minute=end_minute)).astimezone(pytz.utc).isoformat()
        dates_list.append((d.strftime("%Y-%m-%d"), start, end))
    return dates_list


def download_quotes(ticker, start, end, name):
    file_path = f'{QUOTES_FOLDER}/SPY_{name}.pickle'

    if os.path.isfile(file_path):
        print(f'Already downloaded quotes for {name}. Skipping...', flush=True)
        return name

    try:
        start_time = time.time()
        print(f'Invoked {name}')
        alpaca = api.REST(API_KEY, SECRET_KEY)
        quotes = alpaca.get_quotes([ticker], start, end).df
        quotes = quotes.loc[~quotes.index.duplicated(keep='last')]

        if not quotes.empty:
            quotes = quotes[quotes['ask_price'] - quotes['bid_price'] > 0.0]

        quotes.to_pickle(file_path)

        print(f'Completed {name} in {(time.time() - start_time):.1f} seconds. {len(quotes)} total quotes retrieved.', flush=True)
    except Exception as ex:
        print(f'Exception occurred while processing {name}', flush=True)
        print(ex, flush=True)
        traceback.print_exc()
    return name


if __name__ == '__main__':
    dates = generate_dates(START_DATE, END_DATE)

    threads = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=15) as executor:
        for i, date in enumerate(dates):
            date_string = date[0]

            from_iso = date[1]
            to_iso = date[2]

            threads.append(executor.submit(download_quotes, 'SPY', from_iso, to_iso, date_string))

        i = 0
        for completed_thread in as_completed(threads):
            i += 1
            print(f'Status: {i} / {len(threads)}')

            completed_name = completed_thread.result()
            print(f'Completed download of: {completed_name}')

    print(f'Total completed in {(time.time() - start_time):.1f} seconds')
