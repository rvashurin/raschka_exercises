def who():
    print("Alice")

def display(func):
    def inner():
        print("The current user is : ", end="")
        func()
    return inner

if __name__ == "__main__":
    myobj = display(who)
    myobj()
