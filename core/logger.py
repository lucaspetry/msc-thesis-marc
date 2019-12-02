import sys
from datetime import datetime

class Logger(object):

    LOG_LINE = None
    INFO        = '[    INFO    ]'
    WARNING     = '[  WARNING   ]'
    ERROR       = '[   ERROR    ]'
    CONFIG      = '[   CONFIG   ]'
    RUNNING     = '[  RUNNING   ]'
    QUESTION    = '[  QUESTION  ]'

    def log(self, type, message):
        if Logger.LOG_LINE:
            sys.stdout.write("\n")
            sys.stdout.flush()
            Logger.LOG_LINE = None

        sys.stdout.write(str(type) + " " + cur_date_time() + " :: " + message + "\n")
        sys.stdout.flush()

    def log_dyn(self, type, message):
        line = str(type) + " " + cur_date_time() + " :: " + message
        sys.stdout.write("\r\x1b[K" + line.__str__())
        sys.stdout.flush()
        Logger.LOG_LINE = line

    def get_answer(self, message):
        return input(Logger.QUESTION + " " + cur_date_time() + " :: " + message)


def cur_date_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_header(title, length=80, character='='):
    line = length - len(title) - 2
    odd = len(title) % 2 == 1
    prefix = line // 2 + (1 if odd else 0)
    suffix = line // 2
    print(character * prefix, title, character * suffix)
