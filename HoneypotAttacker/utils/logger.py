import time

class SimpleLogger:
    def __init__(self):
        self.start = time.time()

    def log(self, **kwargs):
        msg = f"[{time.time() - self.start:8.2f}s]"
        for k, v in kwargs.items():
            msg += f" {k}={v}"
        print(msg)
