import time
a = [1575766560]
for timeStamp in a:
# timeStamp = 1576297560
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    print(otherStyleTime)
1575663360
1575681240
1575683640
1575692280
1575725400
1575745200
1575747240
1575749400
1575766560
1575787440
1575793080
1575799680
1575812520