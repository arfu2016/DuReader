"""
@Project   : DuReader
@Module    : part_text.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/7/18 11:12 AM
@Desc      : 
"""
# python part_text.py 2 data/wiki.zh.text data/wiki.zh.2.text

import os
import sys
import logging

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 4:
        print("Using: python part_text.py 2 xx.text xx.2.text")
        sys.exit(1)
    inp, outp = sys.argv[2:4]
    line_num = int(sys.argv[1])
    i = 0

    print(inp)

    finput = open(inp, 'r', encoding='utf-8')
    output = open(outp, 'w', encoding='utf-8')
    for line in finput:
        output.write(line)
        i = i + 1
        if i % 2 == 0:
            logger.info("Saved " + str(i) + " articles_seg")
        if i == line_num:
            break

    finput.close()
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
