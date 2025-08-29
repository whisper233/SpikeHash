import logging
import inspect


def get_logger(log_file=None):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False # 好像是spikingjelly库对logging进行了全局配置，如果这里不进行控制会继续向root传播的话，进而被spikingjelly的配置再次输出

    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)
        
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)

    return logger


_logger = get_logger()

def my_print(*args):
    frame = inspect.currentframe().f_back
    abs_path = frame.f_code.co_filename
    file_name = abs_path.split("/")[-1] + ":"
    file_name_c = file_name.center(10)
    _logger.debug("[%s %d]>>[ %s ]" % (file_name_c, frame.f_lineno, " ".join(map(str, args))))
    

__all__ = ["my_print"]

# a = [1,2,3]
# b = ['a', 'b', 'c']
# for i, (c, d) in enumerate(zip(a, b)):
#     print(c)
#     print(d)
#     print('')