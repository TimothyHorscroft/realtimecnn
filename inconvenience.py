"""
    A constant printer is a parameterless function that prints a number.
    For example:

    def f():
        print(5)

    I believe the following is the only method of creating a list of
    100 constant printers, each of which prints a different number.
"""


def constant_printer_generator(num):
    # A function which returns a constant printer function
    return (lambda: print(num))


functions = []
for i in range(100):
    functions.append(constant_printer_generator(i))


# Testing
for f in functions:
    f()
