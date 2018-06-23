"""
Utility functions
"""
import sys
import bisect



def valtoindex(thearray, thevalue, toleft=True):
    if toleft:
        return bisect.bisect_left(thearray, thevalue)
    else:
        return bisect.bisect_right(thearray, thevalue)


def progressbar(thisval, end_val, label='Percent', barsize=60):
    percent = float(thisval) / end_val
    hashes = '#' * int(round(percent * barsize))
    spaces = ' ' * (barsize - len(hashes))
    sys.stdout.write("\r{0}: [{1}] {2:.3f}%".format(label, hashes + spaces, 100.0 * percent))
    sys.stdout.flush()


def primes(n):
    # found on stackoverflow: https://stackoverflow.com/questions/16996217/prime-factorization-list
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def largestfac(n):
    return primes(n)[-1]
