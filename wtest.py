class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0                  # ← 状态存在实例上
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"第 {self.count} 次调用")
        return self.func(*args, **kwargs)

@CountCalls
def greet(): 
    print("hello world")
    return "hello world"
    
greet()
print(greet.count)