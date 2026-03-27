import random

current_min = None
current_max = None

def add_number(x):
    global current_min, current_max
    if current_min is None:
        current_min = x
        current_max = x
    else:
        if x < current_min:
            current_min = x
        if x > current_max:
            current_max = x
    print(f"min={current_min}, max={current_max}")


nums = [random.randint(1, 100) for _ in range(random.randint(5, 20))]
for num in nums:
    add_number(num)