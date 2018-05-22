"""
@Project   : DuReader
@Module    : cartesian_product.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/22/18 11:28 AM
@Desc      : 
"""
import pprint

colors = ['black', 'white']
sizes = ['S', 'M', 'L']
tshirts = [(color, size) for color in colors
                         for size in sizes]

pprint.pprint(tshirts)
