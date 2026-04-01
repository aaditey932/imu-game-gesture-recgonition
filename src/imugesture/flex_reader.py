"""
Flex sensors via ADS1015/ADS1115 (I2C), sharing the bus with MPU6050.
Requires: pip install adafruit-circuitpython-ads1x15

Uses mpu_reader.setup_mpu() first so I2C is initialized once on board.SCL/SDA.
"""

from . import mpu_reader
import adafruit_ads1x15.ads1015 as ADS1015_mod
import adafruit_ads1x15.ads1115 as ADS1115_mod
from adafruit_ads1x15.ads1x15 import Pin
from adafruit_ads1x15.analog_in import AnalogIn

# Must match imugesture.features.N_FLEX_CHANNELS.
N_FLEX_CHANNELS = 2
# Analog inputs (0=A0 … 3=A3) for f0, f1, … — length must equal N_FLEX_CHANNELS.
FLEX_PIN_INDICES = (0, 1)

VALID_GAINS = (2 / 3, 1, 2, 4, 8, 16)
PINS = (Pin.A0, Pin.A1, Pin.A2, Pin.A3)

ads = None
_channels = None

_chip = "1015"
_address = 0x48
_gain = 1.0


def setup_flex(chip="1015", address=0x48, gain=1.0):
    """Initialize ADS1x15 on the shared I2C bus (after MPU). Safe to call multiple times."""
    global ads, _channels, _chip, _address, _gain
    if _channels is not None:
        return
    if len(FLEX_PIN_INDICES) != N_FLEX_CHANNELS:
        raise ValueError(
            "flex_reader: len(FLEX_PIN_INDICES) must equal N_FLEX_CHANNELS"
        )
    if gain not in VALID_GAINS:
        raise ValueError(f"gain must be one of {VALID_GAINS}")
    _chip = chip
    _address = address
    _gain = gain

    mpu_reader.setup_mpu()
    i2c = mpu_reader.i2c

    if chip == "1115":
        ads = ADS1115_mod.ADS1115(i2c, address=address, gain=gain)
    else:
        ads = ADS1015_mod.ADS1015(i2c, address=address, gain=gain)

    _channels = [AnalogIn(ads, PINS[i]) for i in FLEX_PIN_INDICES]
    for c in _channels:
        _ = c.value


def read_flex():
    """One sample per flex channel. Returns tuple of floats (raw ADC counts)."""
    setup_flex()
    return tuple(float(c.value) for c in _channels)
