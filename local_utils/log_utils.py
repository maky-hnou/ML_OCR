"""Set the log config."""
import logging
import os
import os.path as ops
from logging import handlers


def init_logger(level=logging.DEBUG, when='D', backup=7,
                _format='%(levelname)s: %(asctime)s: %(filename)s:%(lineno)'
                        'd * %(thread)d %(message)s',
                datefmt='%m-%d %H:%M:%S'):
    """Initialize log module.

    Parameters
    ----------
    level : str
        The msg to be displayed: logging.(DEBUG/INFO/WARNING/ERROR/CRITICAL).
    when : str
        How to split the log file by time interval
                      'S' : Seconds
                      'M' : Minutes
                      'H' : Hours
                      'D' : Days
                      'W' : Week day
                      default value: 'D'.
    backup : int
        How many backup file to keep default value: 7.
    _format : str
        The format of the log default format:
                       %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d *
                       %(thread)d %(message)s.
    datefmt : str
        The time format.

    Returns
    -------
    str
        A string containing the log.

    """
    formatter = logging.Formatter(_format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    log_path = ops.join(os.getcwd(), 'logs/shadownet.log')
    _dir = os.path.dirname(log_path)
    if (not os.path.isdir(_dir)):
        os.makedirs(_dir)

    handler = handlers.TimedRotatingFileHandler(log_path, when=when,
                                                backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = handlers.TimedRotatingFileHandler(log_path + ".log.wf",
                                                when=when, backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
