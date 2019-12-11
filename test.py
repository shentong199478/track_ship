import time

timeStamp = 1575860520
timeArray = time.localtime(timeStamp)

otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print(otherStyleTime)