"""
@Project   : DuReader
@Module    : mysql_connect.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/14/18 11:40 AM
@Desc      : 
"""
import time
import pymysql


class Mysql:
    conn, cursor = None, None
    connect_count, connect_limit = 0, 2
    params = ["host", "port", "user", "passwd", "db", "charset"]

    def __init__(self, **kwargs):
        try:
            self.conn = pymysql.connect(
                host=kwargs.get("host"),
                port=kwargs.get("port"),
                user=kwargs.get("user"),
                passwd=kwargs.get("passwd"),
                db=kwargs.get("db"),
                charset=kwargs.get("charset", "utf8")
            )
        except pymysql.Error as e:
            self.error_code = e.args[0]
            if self.connect_count < self.connect_limit:
                self.connect_count += 1
                time.sleep(2)
                self.__init__(**kwargs)
            else:
                raise Exception(
                    "Mysql Error!{0}, {1}".format(
                        e.args[0],
                        e.args[1]
                    )
                )
        if kwargs.get('cursor') is True:
            self.cursor = self.conn.cursor(
                cursor=pymysql.cursors.DictCursor)
        else:
            self.cursor = self.conn.cursor()

    def check_init_params(self, **kwargs):
        check_val = True
        for param in self.params:
            if kwargs.get(param) is None:
                check_val = False
                break
        return check_val

    def query(self, sql):
        # self.cursor.execute("SET NAMES utf8")
        res = self.cursor.execute(sql)
        self.conn.commit()
        return res

    def fetchall(self, sql):
        self.query(sql)
        return self.cursor.fetchall()

    def fetchone(self, sql):
        self.query(sql)
        return self.cursor.fetchone()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.cursor.close()
        self.conn.close()

    @staticmethod
    def custom_escape_string(val):
        return pymysql.escape_string(str(val))
