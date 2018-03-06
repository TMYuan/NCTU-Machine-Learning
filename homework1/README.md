# HW01
A program that can do **regularized linear model (polynomial basis) regression**.



You have to do it with both *LSE* and *Newton's method*.



## Input parameters: 

1. the file path and name of a file of which each row represents a data point (common seperated: x,y): 

    1,12

    122,34

    -12,323

     ...

2. the number of polynomial bases n.
3. lambda



## Behavior:



For example, if the number of bases is set as 3, it means that the program is going to find a curve that best fits the data by a x 下標 2 加 b x 下標 1 加 c x 下標 0 等於 y.



Required functions:

a. For LSE:

	1. Use LU decomposition to find the inverse of (ATA + lambda*I), Gauss-Jordan elimination won't be accepted. A is the design matrix.

	2. Print out the equation of the best fitting line and the error.

b. For Newton's method:

	1. Print out the equation of the best fitting line and the error, and compare to LSE.



## NOTE:

* Use whatever programming language you prefer.

* You should use as few functions from any library as possible. That would be great if you implement all detail operations (like matrix operations) by yourself.

* Time complexity is not what we care for now, but if you like to improve it in that regard, it is always good for you.