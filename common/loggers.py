import sys


class BaseLogger(object):
    def __init__(self, output_path):
        self.log = open(output_path, "a")

    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        self.log.flush()
        self.terminal.flush()


class LoggerOut(BaseLogger):
    def __init__(self, output_path):
        super().__init__(output_path)
        self.terminal = sys.stdout


class LoggerErr(BaseLogger):
    def __init__(self, output_path):
        super().__init__(output_path)
        self.terminal = sys.stderr
