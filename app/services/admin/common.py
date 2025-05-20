from datetime import datetime


def getTime(timestamp):
    timestamp_ms = int(timestamp)
    timestamp_sec = timestamp_ms / 1000
    dt = datetime.fromtimestamp(timestamp_sec)
    formatted_date = dt.strftime('%B %d, %Y %H:%M:%S')
    return formatted_date