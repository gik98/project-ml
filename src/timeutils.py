import time

def measure_time(fn, label="", *args, **kwargs):
    start = time.time()
    ret = fn(*args, **kwargs)
    stop = time.time()

    print("Function {} took {} s.".format(label, stop - start))
    return ret