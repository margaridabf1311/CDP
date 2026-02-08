# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:12:43 2022

@author: Luís Silveira Santos
"""

# FIRST INTERACTION
print("Hello, World!")
string = "Hello, World!"
print(string)


# VARIABLE TYPES AND FORMATS
# String
name = "Ana"
print(name)

# Integer
age = 20
print(age)

# Float
age_new = 20.0
height = 170.5
print(height)

# Boolean
is_student = True
print(is_student)


# INPUT AND OUTPUT
yname = input("Como te chamas? ")
print(f"Olá, {yname} :-)")


# CONDITIONALS
age = 17
if age >= 18:
    print("Parabéns, és maior de idade.")
else:
    print("Nada feito, és menor de idade.")


# LOOPS
# For loop
for i in range(1,1000000):
    print(i)  # Prints 1, 2, 3, 4, ..., n-1

# While loop
count = 0
while count <= 5:
    print(count)
    count += 1


# FUNCTIONS
def greet(name):
    return f"Olá, {name}!"
print(greet("Ana"))


# BASIC OPERATIONS
# Addition
a = 5 + 3
print(a)

# Subtraction
b = 10 - 4  
print(b)

# Multiplication
c = 7 * 2   
print(c)

# Division
d = 8 / 2  
print(d)

# Floor Division (rounds down)
e = 8 // 3 
print(e)

# Modulus (remainder)
f = 8 % 3 
print(f)

# Exponentiation (power)
g = 2 ** 3
print(g)

import math
result = math.sqrt(25)
print(result)
result_alt = 25 ** (1/2)
print(result_alt)

import random
random_number = random.randint(1, 10)
print(random_number)

import datetime
now = datetime.datetime.now()
print(now)


# COMPARISON BOOLEANS
x = 5
y = 3

# Equal to
print(x == y) 

# Not equal to
print(x != y) 

# Greater than
print(x > y) 

# Less than
print(x < y) 

# Greater than or equal to
print(x >= y) 

# Less than or equal to
print(x <= y) 


# LOGICAL BOOLEANS
a = True
b = False

# and 
print(a and b) 

# or
print(a or b) 

# not
print(not a) 


# ASSIGNMENT OPERATIONS
x = 5       # Assigns 5 to x
print(x)
x += 2      # Adds 2 to x, now x = 7
print(x)
x -= 3      # Subtracts 3 from x, now x = 4
print(x)
x *= 2      # Multiplies x by 2, now x = 8
print(x)
x /= 4      # Divides x by 4, now x = 2.0
print(x)
x **= 3     # x raised to the power of 3, now x = 8.0
print(x)


# STRING OPERATIONS
str1 = "Hello"
str2 = "World"

# Concatenation
result = str1 + " " + str2   # "Hello World"
print(result)

# Repetition
result = str1 * 3   # "HelloHelloHello"
print(result)


# MEMBERSHIP AND IDENTITY OPERATIONS
# Membership
list1 = [1, 2, 3, 4]
print(2 in list1) 
print(5 not in list1)
print([2,5] in list1)

# Identity
x = [1, 2, 3]
y = x
z = [1, 2, 3]
print(x is y)
print(x is z)


# MATRIX OPERATIONS

import numpy as np

# Creating a 2x2 matrix
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Creating a 2x3 matrix
C = np.array([[9, 10, 11], [12, 13, 14]])
D = np.array([[15, 16, 17], [18, 19, 20]])

# Matrix addition
E = A + B 
print(E)

# Matrix subtraction
F = A - B  
print(F)

# Matrix multiplication using @ (dot product)
G = A @ B
print(G)

# Alternatively, you can use np.dot()
H = np.dot(A, B)
print(H)

# Element-wise multiplication
I = A * B
print(I)

# Matrix transpose
A_transposed = A.T
print(A_transposed)

# Matrix determinant
det_A = np.linalg.det(A)
print(det_A)

# Matrix inversion
A_inv = np.linalg.inv(A)
print(A_inv)

# Matrix shape and size
print(A.shape)
print(A.size) 

# Matrix scalar multiplication
J = 3 * A
print(I)

# Matrix eigenstructure
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)
print(eigenvectors)

# Matrix rank
rank = np.linalg.matrix_rank(A)
print(rank)


# SYMBOLIC CALCULUS #
import sympy as sp

# Define symbolic variables
x, y = sp.symbols('x y')

# Differentiation
f = x**2 + 3*x + 2
df = sp.diff(f, x)  # Differentiate with respect to x
print(df)
d2f = sp.diff(f, x, 2)  # Second derivative
print(d2f)

# Integration
integral = sp.integrate(f, x)
print(integral)
definite_integral = sp.integrate(f, (x, 0, 2))
print(definite_integral)

# Solving equations
eq = x**2 - 5*x + 6 
solutions = sp.solve(eq, x)
print(solutions)

# Simplification and expansion (+ trigonometric functions)
expr = (x + 1)**2
simplified = sp.simplify(expr)
expanded = sp.expand(expr)
print(simplified, "and", expanded)
trig_expr = sp.sin(x)**2 + sp.cos(x)**2
simplified_trig = sp.simplify(trig_expr)
print(simplified_trig)

# Limits
f = 1 / x
lim1 = sp.limit(f, x, 0)  # Limit as x approaches 0
lim2 = sp.limit(f, x, math.inf) # Limit as x approaches +infinity
print(lim1)
print(lim2)

# Taylor Series expansion
taylor_expansion_1 = sp.series(sp.sin(x), x, 0, 6)  # Around x=0, up to 6th order
taylor_expansion_2 = sp.series(sp.exp(x), x, 0, 6)  # Around x=0, up to 6th order
print(taylor_expansion_1)
print(taylor_expansion_2)


# Plotting
f = x**2 + 3*x + 2
sp.plot(f,xlim=(-5,5),ylim=(-5,5))

