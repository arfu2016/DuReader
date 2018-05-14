"""
@Project   : DuReader
@Module    : mysql_settings.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/14/18 1:42 PM
@Desc      :
mysql_db_robot:
host: "120.55.102.1"
user: "u_all"
passwd: "hbhb0617"
port: 7788
db: "db_robot"
db_name: "db_robot"
db_name_news: "db_robot"
db_name_regex: "db_robot"
db_name_question_log: "db_robot"
db_name_match: "db_robot"
"""
import logging
import os
import sys
import pprint
import pickle
from .mysql_connect import Mysql

robot_user_interact_log = {"host": "120.55.102.1", "port": 7788,
                           "user": "u_all", "passwd": "hbhb0617",
                           "db": "db_robot"
                           }


def set_logger():
    # for logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    return logger


def test():
    logger = set_logger()
    interact_log = Mysql(**robot_user_interact_log)
    log_number = interact_log.fetchall(
        "select count(*) from robot_user_interact_log")
    logger.info('There are {} records'.format(log_number[0][0]))

    log_columns = interact_log.fetchall(
        "describe robot_user_interact_log")
    column_names = [log_column[0] for log_column in log_columns]
    log_example = interact_log.fetchone(
        "select * from robot_user_interact_log")
    # print(log_example)
    pprint.pprint(list(zip(column_names, log_example)))
    log_templates = interact_log.fetchall(
        "select template from robot_user_interact_log")
    log_tpl_yes = [tpl[0] for tpl in log_templates if tpl[0]]
    log_tpl_unique = list(set(log_tpl_yes))
    logger.info('There are {} valid templates'.format(len(log_tpl_yes)))
    logger.info('There are {} unique templates'.format(len(log_tpl_unique)))
    pprint.pprint(log_tpl_unique[0:10])


def save_data():
    logger = set_logger()
    interact_log = Mysql(**robot_user_interact_log)

    log_data = interact_log.fetchall(
        "select template, intent_name from robot_user_interact_log")
    # print(log_data[0])
    log_data_yes = [(tpl, intent) for tpl, intent in log_data if tpl]
    log_data_unique = list(set(log_data_yes))
    logger.info(
        'There are {} unique templates and intents'.format(
            len(log_data_unique)))

    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, 'data/templates_intents.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(log_data_unique, f)
