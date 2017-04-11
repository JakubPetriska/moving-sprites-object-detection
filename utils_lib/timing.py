import time


def start_timer():
    return time.time()


def get_duration_secs(start):
    return round((time.time() - start), 1)


def get_duration_minutes(start):
    return round((time.time() - start) / 60, 1)
