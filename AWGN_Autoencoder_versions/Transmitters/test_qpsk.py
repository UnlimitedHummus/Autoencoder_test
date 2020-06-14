import unittest
from .qpsk import QPSKTransmitter
import numpy as np


class TestTransmitByte(unittest.TestCase):
	transmitter = QPSKTransmitter()

	def test_transmission(self):
		# Transmit an integer value
		input_int = np.uint8(100)
		output_int = self.transmitter.transmit_byte(input_int)
		# Make sure only uint8 integers are returned
		self.assertIsInstance(output_int, np.uint8, "Must return uint8 integer")
		# Make sure input and output match
		self.assertEqual(
			output_int, input_int,
			"Transmitted integer is different from original integer"
		)

	def test_accepted_types(self):
		# Make sure only np.uint8 integers are accepted as parameters
		self.assertRaises(TypeError, self.transmitter.transmit_byte, int(1))
		self.assertRaises(TypeError, self.transmitter.transmit_byte, np.int64(1))
		self.assertRaises(TypeError, self.transmitter.transmit_byte, '1')


class TestTransmitBitstream(unittest.TestCase):
	transmitter = QPSKTransmitter()

	def test_transmission(self):
		input_bitstream = np.array([1, 0, 0, 1], dtype=np.uint8)
		output_bitstream = self.transmitter.transmit_bitstream(input_bitstream)
		# Make sure only flat numpy.ndarrays with numpy.uint8 values are returned
		self.assertEqual(output_bitstream.dtype, np.dtype(np.uint8), "Must return array of uint8 values")
		self.assertEqual(len(output_bitstream.shape), 1, "Must return flat array")
		# Make sure input and output match
		np.testing.assert_array_equal(
			output_bitstream, input_bitstream,
			"Transmitted bitstream is different from original bitstream"
		)

	def test_accepted_types(self):
		# Make sure only flat numpy.ndarrays with even numbers of numpy.uint8 values are accepted as parameters
		self.assertRaises(TypeError, self.transmitter.transmit_bitstream, int(1))
		self.assertRaises(TypeError, self.transmitter.transmit_bitstream, np.array([0, 1, 0, 0]))
		self.assertRaises(TypeError, self.transmitter.transmit_bitstream, np.array([0, 1, 0, [1, 0]]))
		self.assertRaises(TypeError, self.transmitter.transmit_bitstream, np.array([1, 1, 0], dtype=np.uint8))
		# self.assertRaises(ValueError, self.transmitter.transmit_bitstream, np.array([10, 12], dtype=np.uint8))


class TestTransmitByteArray(unittest.TestCase):
	transmitter = QPSKTransmitter()

	def test_transmission(self):
		input_array = np.array([0, 100, 233, 12], dtype=np.uint8)
		output_array = self.transmitter.transmit_byte_array(input_array)
		# Make sure only flat numpy.ndarrays with numpy.uint8 values are returned
		self.assertEqual(output_array.dtype, np.dtype(np.uint8), "Must return array of uint8 values")
		self.assertEqual(len(output_array.shape), 1, "Must return flat array")
		np.testing.assert_array_equal(
			output_array, input_array,
			"Transmitted array is different from original array"
		)

	def test_accepted_types(self):
		self.assertRaises(TypeError, self.transmitter.transmit_byte_array, int(1))
		self.assertRaises(TypeError, self.transmitter.transmit_byte_array, np.array([0, 17, 0, 11]))
		self.assertRaises(TypeError, self.transmitter.transmit_byte_array, np.array([0, 100, 20, [10, 0]]))
