"""
@Project   : DuReader
@Module    : try.py
@Author    : Deco [deco@cubee.com]
@Created   : 3/16/18 5:13 PM
@Desc      : https://docs.python.org/2/library/argparse.html
"""

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
parser.add_argument('--prepare', action='store_true')

args = parser.parse_args()
if args.prepare:
    print(sum(args.integers))

print(args.accumulate(args.integers))
print(args.integers)
print(args.prepare)
print(args.accumulate)
print(args.sum)

# python try.py 1 2 --sum
# python try.py 1 2 --prepare
