"""
# This is a demo to learn python functional programming.

"""
import numpy as np


def run(obj):
	return obj.read()


class Variable:
	def __init__(self, init_value=0):
		def func():
			# print('Variable')
			return init_value
		self.__func = func

	def read(self):
		return self.__func()

	def assign(self, value):
		def func():
			return value
		self.__func = func


class Add:
	def __init__(self, obj1, obj2):
		def func():
			return obj1.read() + obj2.read()
		self.__func = func

	def read(self):
		# print('Add')
		return self.__func()


class Multiply:
	def __init__(self, obj1, obj2):
		def func():
			return obj1.read() * obj2.read()
		self.__func = func

	def read(self):
		# print('Multiply')
		return self.__func()


def my_sum(data_list):
	result = 0
	for item in data_list:
		result = result + abs(item)
	return result/1000


weight = Variable(6)
bias = Variable(0)

x_holder = Variable(0)
y_holder = Add(Multiply(weight, x_holder), bias)


x_train_data = np.linspace(-1, 1, 1000, dtype=np.float32)
y_train_data = 2.5*x_train_data


x_holder.assign(x_train_data)

for i in range(50):
	origin_weight = run(weight)
	origin_bias = run(bias)

	error1 = my_sum(run(y_holder)-y_train_data)
	weight.assign(origin_weight + 0.1)
	error2 = my_sum(run(y_holder)-y_train_data)

	if error1 < error2:
		weight.assign(origin_weight - 0.1)

	print(
		# i,
		'error:', my_sum(run(y_holder) - y_train_data),
		'weight:', run(weight),
	)








