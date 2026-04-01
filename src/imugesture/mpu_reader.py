"""
MPU6050 reader for gesture recognition (I2C).
Requires: pip install adafruit-circuitpython-mpu6050
"""

i2c = None
mpu = None


def setup_mpu():
    """Initialize I2C and MPU6050 once."""
    global i2c, mpu
    if mpu is not None:
        return
    import board
    import busio
    import adafruit_mpu6050
    i2c = busio.I2C(board.SCL, board.SDA)
    mpu = adafruit_mpu6050.MPU6050(i2c)


def read_accel():
    """Read one accelerometer sample. Returns (ax, ay, az) in m/s²."""
    setup_mpu()
    return mpu.acceleration


def read_gyro():
    """Read one gyroscope sample. Returns (gx, gy, gz) in rad/s."""
    setup_mpu()
    return mpu.gyro
