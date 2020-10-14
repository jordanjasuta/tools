"""
test script for automatic updates
"""

from datetime import datetime, timedelta
from datetime import time as dt_time
import time

shutdown_time = dt_time(2,2,0)
print('.....shutdown time: ', shutdown_time)

current_time = datetime.now()
print('.....current time: ', current_time)
date_started = current_time - timedelta(days=1)
print('.....date started: ', date_started)


list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q']

for letter in list:
    # wait 2 seconds
    time.sleep(2)
    print(letter)
    current_time = datetime.now()
    print('.....current time: ', current_time)
    if current_time.time() > shutdown_time and date_started.day < current_time.day:
        break
