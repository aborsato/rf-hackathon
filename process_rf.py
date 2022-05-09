import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
from timeit import default_timer as timer   # library to check time to run program
from numba import vectorize     #library to use GPU

# for vectorizing need to add:
#   @vectorize(['float32(float32,float32)'],target='cuda')


starttime = timer()
# use "linspace" function to create even spaced values according to sample rate and duration
def create_t(duration, sample_rate):
    return np.linspace(0, duration, sample_rate * duration, endpoint=False)


# create sin wave with duration and frequency
def create_sin(x, freq):
    frequencies = x * freq
    # 2pi because np.sin takes radians
    return np.sin((2 * np.pi) * frequencies)


def create_pdf(x, mu, std):
    return stats.norm.pdf(x, mu, std) 

def moving_avg(x, n):
    d = pd.Series(x)
    return d.rolling(n, min_periods=1, center=True).mean().fillna(0.0).to_numpy()

# normalize an array to 0:1
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# add random noise to wanted signal
def random_noise(x, factor=1):
    return np.random.random(len(x)) * factor

# normalize random noise??
def random_pdf(x):
    xmax = max(x)
    mu = np.random.rand() * xmax
    std = max(0.1, min(np.random.random(), 0.9)) * xmax
    return normalize(create_pdf(x, mu, std))


def read_signal(frequency, duration, chunks):
    sample_rate = frequency * 10            # rate at which sine wave will be sampled
    x = create_t(duration, sample_rate)     # create variable "x" to represent signal duration
    signal_strength = (random_pdf(x) + random_pdf(x))
    y_noise = random_noise(x, 10)
    y_signal = create_sin(x, frequency) * signal_strength + y_noise

    df = pd.DataFrame()
    df['x'] = x
    df['signal'] = y_signal

    # split the dataframe in chunks of equal size
    def f(ys, freq_position=None):
        frequencies = np.abs(rfft(ys.to_numpy()))  # use right FFT to avoid using negative numbers.
        return frequencies[freq_position[0]:freq_position[1]].max()

    # find where is the frequency inside each chunk's array
    NC = int(sample_rate * duration / chunks)
    xf = rfftfreq(NC, 1 / sample_rate)
    FREQUENCY_THRESHOLD = int(frequency * 0.05)
    idx = np.argwhere(abs(xf - frequency) < FREQUENCY_THRESHOLD)
    freq_position = (idx.min(), idx.max())

    
    # apply rfft for each chunk
    df = df.groupby(np.arange(len(df)) // int(len(df) / chunks))['signal'].apply(f, freq_position=freq_position).reset_index()

    # for some reason, the result of rfft is proportional to the power
    df['signal'] = (df['signal'] / df['signal'].max()) * np.random.randint(0, 100)
    return df
    

def print_message():
    print("                                           ___ ")
    print("                                     |     | | ")
    print("                                    / \    | | ")
    print("     PROCESSING...                 |--o|===|-| ")
    print("                                   |---|   | | ")
    print("                                  /     \  | | ")
    print("     /\                          | N     | | | ")
    print("    /  \    _____   _ _ __ ___   | A     |=| | ")
    print("   / /\ \  |_  / | | | '__/ _ \  | S     | | | ")
    print("  / ____ \  / /| |_| | | |  __/  |_A_____| |_| ")
    print(" /_/    \_\/___|\__,_|_|  \___|   |@| |@|  | | ")
    print("                                ___________|_|_")

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='RF Generator.')
parser.add_argument('--iterations', help='number of iterations', type=int, default=1)
parser.add_argument('--duration', help='duration (s) for each iteration', type=int, default=60)
args = parser.parse_args()

print_message()

SIGNAL_THRESHOLD = 30 # any number from 0 to 100
CHUNKS = 10

records = []
start = pd.Timestamp.utcnow()
process_start = start.to_pydatetime()
for i in range(args.iterations):
    frequency = np.random.choice([3400, 9000, 12000], size=None, p=[0.6, 0.2, 0.2])  # Weighted choise 60% time the first one
    print(f'Iteration {i}, freq={frequency}')
    dff = read_signal(frequency, args.duration, CHUNKS)

    # restore timestamp
    end = start + pd.DateOffset(seconds=args.duration)

    dff['time'] = pd.to_datetime(np.linspace(start.value, end.value, CHUNKS))
    start = end

    dff = dff[dff['signal'] > SIGNAL_THRESHOLD]
    if len(dff) == 0:
        continue

    record = dict(
        ts_start = dff.time.min().to_pydatetime(),
        ts_max = dff.time[dff.signal.idxmax()].to_pydatetime(),
        ts_end = dff.time.max().to_pydatetime(),
        frequency = frequency,
        power_min = dff.signal.min(),
        power_max = dff.signal.max(),
        power = dff.signal.median(),
    )
    records.append(record)


df = pd.DataFrame(records)
# save the data in different formats
file_name = f'o{process_start.strftime("%y%m%d%H%M%S")}'
df.to_parquet(f'{file_name}.parquet', index=False, allow_truncated_timestamps=True)
df.to_csv(f'{file_name}.csv', index=False, date_format='%Y-%m-%dT%H:%M:%S.%f%z')

duration = timer() - starttime
print ("time taken", duration)