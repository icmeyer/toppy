import numpy as np
import pandas as pd

def import_csv(filename, tallytype='dose', dimensions=[]):
    """
    Parameters
    ----------
    tallytype : str
        one of (dose, DVH)
    """
    # Get header information
    header = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line[0] != '#':
                break
            header.append(line)
    # Get column values
    if tallytype == 'dose':
        name, columns = header[-1].split(':')
        columns = columns.strip().split()
        name = name[2:-1]
    elif tallytype == 'DVH':
        columns = header[-1].split(',')
        columns = [c.strip() for c in columns]
        name = 'DVH'
    columns = np.hstack([dimensions, columns])
    print(columns)
    df = pd.read_csv(filename, names=columns, skiprows=len(header))
    df.attrs['scorer_name'] = name
    df.attrs['header'] = header
    return df
