# This transmitter uses QPSK modulation to transmit a byte, an array of bytes or an array of bits
# over a channel with 'added white gaussian noise' (AWGN). All arrays must use np.uint8 integers.

# import logging
import numpy as np
from komm import PSKModulation, AWGNChannel, int2binlist, binlist2int


# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


class QPSKTransmitter:
	"""The QPSKTransmitter class provides methods for transmitting integers and 1D arrays of bits or integers."""

	def __init__(self, snr: float = np.inf):
		self.channel = AWGNChannel(snr=snr)
		self.qpsk = PSKModulation(4, phase_offset=np.pi / 4)

	def set_snr(self, snr: float = np.inf) -> None:
		self.channel.snr = snr

	# TRANSMISSION FUNCTIONS

	def transmit_byte(self, input_int: np.uint8) -> np.uint8:
		"""Takes a uint8 integer value, modulates it via QPSK, transmits it over the channel
		and converts it back to a uint8 integer value."""

		if np.dtype(input_int) is not np.dtype(np.uint8):
			raise TypeError("The integer parameter must be of type uint8")

		# Modulate
		modulated = self.qpsk.modulate(int2binlist(input_int, 8))
		# Transmit over the channel channel with 'additive white gaussian noise'
		channel_out = self.channel(modulated)
		# Demodulate
		demodulated = self.qpsk.demodulate(channel_out)
		# Return the received integer
		return np.uint8(binlist2int(demodulated))

	def transmit_bitstream(self, input_bitstream: np.ndarray) -> np.ndarray:
		"""Takes a uint8 bitstream array, modulates it via QPSK, transmits it over the channel
		and converts it back to a bitstream."""

		if type(input_bitstream) is not np.ndarray:
			raise TypeError("The input bitstream must be a numpy array")
		if input_bitstream.dtype is not np.dtype(np.uint8):
			raise TypeError("The input bitstream must consist of uint8 integers")
		if len(input_bitstream) % 2 is not 0:
			raise TypeError("The input array must have an even number of bits")

		# Modulate
		modulated = self.qpsk.modulate(input_bitstream)
		# Transmit over the channel channel with 'additive white gaussian noise'
		channel_out = self.channel(modulated)
		# Demodulate
		demodulated = np.array(self.qpsk.demodulate(channel_out), dtype=np.uint8)

		return demodulated

	def transmit_byte_array(self, input_array: np.ndarray) -> np.ndarray:
		"""Takes a 1D array of uint8 bytes, transmits it and converts it back to a 1D array."""

		# Allocate an output array of correct size
		number_of_ints = len(input_array)
		output_array = np.zeros(number_of_ints, dtype=np.uint8)

		if type(input_array) is not np.ndarray:
			raise TypeError("The input array must be a numpy array")
		if input_array.dtype is not np.dtype(np.uint8):
			raise TypeError("The input array must consist of uint8 integers")
		if len(input_array.shape) is not 1:
			raise TypeError("The input array must be of flat shape")

		# Transmit the array item-wise
		for i in range(number_of_ints):
			output_array[i] = self.transmit_byte(input_array[i])
		return output_array
