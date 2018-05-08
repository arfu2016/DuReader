"""
@Project   : DuReader
@Module    : set_logger.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/8/18 5:59 PM
@Desc      : 
"""
import logging
import os
import sys


class SentenceTarget:

    def __init__(self, original_data=None):

        self._set_logger()

    def _set_logger(self):

        self.logger = logging.getLogger("vector")
        # self.logger.setLevel(logging.ERROR)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)


def set_logger():
    # for logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    return logger
