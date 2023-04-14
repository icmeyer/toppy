import numpy as np
import pandas as pd

def import_csv(filename, dimensions):
    # Get header information
    header = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line[0] != '#':
                break
            header.append(line)
    # Get column values
    name, columns = header[-1].split(':')
    name = name[2:-1]
    columns = columns.strip().split()
    columns = np.hstack([dimensions, columns])
    print(columns)
    df = pd.read_csv(filename, names=columns, skiprows=len(header))
    df.attrs['scorer_name'] = name
    df.attrs['header'] = header
    return df
