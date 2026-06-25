

def blackbox(x):
    n=len(x)
    if n % 3 == 0:
        return 0
    if n % 5 == 0:
        return 1
    else:
        return int(x[n // 2])  