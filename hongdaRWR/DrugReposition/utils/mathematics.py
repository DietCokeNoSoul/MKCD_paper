from typing import *

def bitlist(x: int, reverse: bool = True, fix_length: int = 0) -> List[int]:
    '''返回整数`x`的二进制数组
    \n x
    \n reverse
    \n fix_length: 为0时去掉前导0，大于0时数组长度为`max(fix_length,len(res))`
    '''
    res = list()
    while x != 0:
        res.append(1 if x & 1 == 1 else 0)
        x = x//2
    for _ in range(max(fix_length-len(res), 0)):
        res.append(0)
    if reverse:
        res.reverse()
    return res
