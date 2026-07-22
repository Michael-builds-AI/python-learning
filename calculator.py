num1 = int(input("Enter first number:"))
num2 = int(input("Enter seond number:"))

operation = input ("choose (+,-,/,*):")
if operation == "+":
    print(num1+num2)
elif operation == "-":
    print(num1-num2)
elif operation == "*":
    print(num1*num2)
elif operation == "/":
    print(num1/num2)
else:
    print("invalid operation")

