def my_decor(n):
    def dec(func):
        def wrapper(*args, **kwargs):
            print("Before")
            for _ in range(n):
                func(*args, **kwargs)
            print("After")
        return wrapper
    return dec

@my_decor(3)
def my_func():
    print("Inside")
    
my_func()