"""
@Project   : DuReader
@Module    : tf_logger.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/17/18 6:26 PM
@Desc      : 
"""
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# 1压制info，只输出warn以上级别

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)
