"""about data"""


def get_time_status(ds):
    """

    :param ds:
    """
    if ds == 'support':
        return 'time', 'dead'
    elif ds == 'metabric':
        return 'time', 'status'
    elif ds == 'flchain':
        return 'futime', 'death'
    elif ds == 'gbsg2':
        return 'time', 'cens'
    elif ds == 'whas500':
        return 'lenfol', 'status'
    else:
        print('Data not included!')
