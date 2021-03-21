import time
import math
import IMU
import datetime
import os
import sys


IMU.detectIMU()     #Detect if BerryIMU is connected.
if(IMU.BerryIMUversion == 99):
    print(" No BerryIMU found... exiting ")
    sys.exit()
IMU.initIMU()       #Initialise the accelerometer, gyroscope and compass


a = datetime.datetime.now()


def get_movement(a=a):
    #Read the accelerometer
    ACCx = IMU.readACCx()
    ACCy = IMU.readACCy()
    ACCz = IMU.readACCz()

    ##Calculate loop Period(LP). How long between Gyro Reads
    b = datetime.datetime.now() - a
    a = datetime.datetime.now()
    LP = b.microseconds/(1000000*1.0)
    # outputString = "Loop Time %5.2f " % ( LP )

    accnorm = math.sqrt(ACCx * ACCx + ACCy * ACCy + ACCz * ACCz)
    # outputString += " Acceleration: "+str(accnorm)
    if abs(accnorm - 8500) >= 1000:
        return True
    return False
    # print(outputString)

if __name__ == "__main__":
    while True:
        get_movement()
        #slow program down a bit, makes the output more readable
        time.sleep(0.3)