import math

def softmax(values, temperature=1.0):
    if not values or temperature <= 0:
        raise ValueError("Nonempty values and positive temperature required")
    maximum = max(values)
    weights = [math.exp((x - maximum) / temperature) for x in values]
    total = sum(weights)
    return [w / total for w in weights]