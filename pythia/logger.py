from datetime import datetime

import pytz


class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.log_stream = None

    def info(self, msg):
        self.out("[{}][INFO] {}".format(datetime.now(tz=pytz.utc), msg))

    def error(self, msg):
        self.out("[{}][ERROR] {}".format(datetime.now(tz=pytz.utc), msg))

    def __enter__(self):
        if self.log_file is not None:
            self.log_stream = open(self.log_file, 'a+')
            self.out = self.file_writer
        else:
            self.out = print
        return self

    def file_writer(self, msg):
        self.log_stream.write(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_stream is not None:
            self.log_stream.close()
