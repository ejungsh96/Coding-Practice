# from math import *

# name = "John"
# age = 35

# # print("My name is " + name + ", ")
# # print("and I'm " + age + "years old.")

# # print(len(name))
# # print(len(age))

# print(3 + 4 * 5)

# name = input("Enter your name: ")
# age = input("Enter your age: ")
# print("Hello " + name + "! You are " + age)

# num1 = input("Enter a number: ")
# num2 = input("Enter another number: ")
# result = float(num1) + float(num2)

# print(result)

# friends = ["Kevin", "Sean", "Jim", "Oscar", "Toby"]

# print(friends[:-1])

# --------

# secret_number = 9
# guess_count = 0
# guess_limit = 3
# while guess_count < guess_limit:
#     guess = int(input('Guess: '))
#     guess_count += 1
#     if guess == secret_number:
#         print('You won!')
#         break
# else:
#     print('Sorry, you failed')

# ---------

# numbers = [5, 2, 5, 2, 2]

# # for x in numbers:
# #     for y in range(x):
# #         print('x', end ='')
# #     print("")

# for x in numbers:
#     output = ''
#     for y in range(x):
#         output += 'x'
#     print(output)

# ------------

# list = [1, 2, 3, 4, 5, 6, 15, 10]

# biggest = 0
# for i in list:
#     if i > biggest:
#         biggest = i
# print(biggest)
    
# list_1 = [1, 2, 3]
# list_2 = list_1.copy()
# print(f'list_2: {list_2}')
# list_1.append(10)
# print(f'list_2 after list_1 is append: {list_2}')


# list = [1, 1, 2, 3, 4, 5, 5, 5, 6, 7]
# # number_count = []
# # for i in list:
# #     list.count(i)
# #     number_count.append(i)
# # index = 0
# # for j in number_count:
# #     if j > 1:
# #         list.remove(list.find(index))
# #         index += 1
# unique = []
# for number in list:
#     if number not in unique:
#         unique.append(number)
# print(unique)


# tuple = (1, 2, 3, 3, 4)
# print(tuple)

# number = {
#     "1": "One",
#     "2": "Two",
#     "3": "Three",
#     "4": "Four",
#     "5": "Five",
#     "6": "Six",
#     "7": "Seven",
#     "8": "Eight",
#     "9": "Nine",
#     "10": "Ten"
# }

# input_number = input("Telephone number: ")
# output = ""
# for i in input_number:
#     output += number.get(i, "invalid number") + " "
# print(output)

# class Point():
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def move(self):
#         print("move")
    
#     def draw(self):
#         print("draw")

# point = Point(10, 20)

# class Mammal:
#     def walk(self):
#         print("walk")

# class Dog(Mammal):
#     def bark(self):
#         print("bark")

# class Cat(Mammal):
#     pass

# from pathlib import Path

# path = Path()
# for file in path.glob('*'):
#     print(file)


# def print_stars(n):
#     for i in range(1, n + 1):
#         for j in range(i):
#             print('*', end='')
#         print('')

# number = int(input('number: '))
# print_stars(number)

# def print_stars_triangle(n):
#     for i in range(n, 0, -1):
#         for j in range(n -i):
#             print(' ', end='')
#         for j in range(2 * i - 1):
#             print('*', end='')
#         print('')

# number = int(input('number: '))
# print_stars_triangle(number)

# list = [1, 2, 3] + [5, 6, 7]
# print(list)


# def count_characters_in_string(string):
#     output = {}
#     for char in string:
#         if char in output:
#             output[char] += 1
#         else:
#             output[char] = 1
    
#     return output

# print(count_characters_in_string("character"))



# Counter, namedtuple, OrderedDict, defaultdict, deque

# from collections import Counter
# a = "aaaaaaaaaabbbbbbbbbbcccccccccc"
# my_counter = Counter(a)
# print(my_counter.most_common(1))

# from collections import namedtuple
# Point = namedtuple('Point', 'x, y')
# pt = Point(1, -4)
# print(pt)
# print(pt.x, pt.y)

# from collections import OrderedDict
# ordered_dict = OrderedDict()
# ordered_dict['a'] = 1
# ordered_dict['b'] = 2
# ordered_dict['c'] = 3
# ordered_dict['d'] = 4
# print(ordered_dict)

# from collections import defaultdict
# d = defaultdict(int)
# d['a'] = 1
# d['b'] = 2
# d['c'] = 3
# d['d'] = 4
# print(d['x'])

# from collections import deque
# deq = deque()
# deq.append(1)
# deq.append(2)
# deq.append(3)
# deq.appendleft(0)
# print(deq)
# deq.pop()
# deq.popleft()
# print(deq)
# deq.extend([10, 11, 12])
# print(deq)
# deq.extendleft([6, 7, 8]) # added in reversed order
# print(deq)
# deq.rotate(1)
# print(deq)


# # Lambda
# points_2D = [(1, 2), (15, 1), (5, -1), (10, 4)]
# points_2D_sorted = sorted(points_2D, key=lambda x: x[0] + x[1])

# print(points_2D)
# print(points_2D_sorted)

# # map(func, seq)
# a = [1, 2, 3, 4, 5]
# b = map(lambda x: x * 2, a)
# print(list(b))

# # list comprehension syntax
# c = [x * 2 for x in a]
# print(c)

# # filter(func, seq)
# a = [1, 2, 3, 4, 5, 6]
# b = filter(lambda x: x % 2 == 0, a)
# print(list(b))

# # list comprehension syntax
# c = [x for x in a if x % 2 == 0]
# print(c)

# # reduce(func, seq)
# from functools import reduce
# a = [1, 2, 3, 4, 5, 6]

# product_a = reduce(lambda x, y: x * y, a)
# print(product_a)



# # Errors and Exceptions
# x = -5
# # if x < 0:
# #     raise Exception('x should be positive')

# assert(x >= 0), 'x is not positive' # assert: '확인', '표명' 정도로 해석

# try:
#     a = 5 / 0
#     b = a + '10'
# except ZeroDivisionError as e:
#     print(e)
# except TypeError as e:
#     print(e)
# else:
#     print('Everything is fine.')
# finally:    # run anyway
#     print('cleaning up...')

# class ValueTooHighError(Exception):
#     pass

# class ValueTooSmallError(Exception):
#     def __init__(self, message, value):
#         self.message = message
#         self.value = value

# def test_value(x):
#     if x > 100:
#         raise ValueTooHighError('value is too high')
#     if x < 5:
#         raise ValueTooSmallError('value is too small', x)

# try:
#     test_value(1)
# except ValueTooHighError as e:
#     print(e)
# except ValueTooSmallError as e:
#     print(e.message, e.value)


