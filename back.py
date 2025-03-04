def tanh(x):
    return (2 / (1 + (2.718281828459045 ** (-2 * x)))) - 1  

def tanhder(x):
    return 1 - tanh(x) ** 2  

w1, w2 = 0.15, 0.2   
w3, w4 = 0.25, 0.3   
w5, w6 = 0.4, 0.45   
w7, w8 = 0.5, 0.55   

b1, b2 = 0.5, 0.7    

i1, i2 = 0.05, 0.10  
target_o1, target_o2 = 0.01, 0.99  

lr = 0.5  

net_h1 = w1 * i1 + w2 * i2 + b1 * 1
net_h2 = w3 * i1 + w4 * i2 + b1 * 1

out_h1 = tanh(net_h1)
out_h2 = tanh(net_h2)

net_o1 = w5 * out_h1 + w6 * out_h2 + b2 * 1
net_o2 = w7 * out_h1 + w8 * out_h2 + b2 * 1

out_o1 = tanh(net_o1)
out_o2 = tanh(net_o2)

error_o1 = 0.5 * (target_o1 - out_o1) ** 2
error_o2 = 0.5 * (target_o2 - out_o2) ** 2

total_error = error_o1 + error_o2

delta_o1 = (target_o1 - out_o1) * tanhder(net_o1)
delta_o2 = (target_o2 - out_o2) * tanhder(net_o2)

delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * tanhder(net_h1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * tanhder(net_h2)

w5 += lr * delta_o1 * out_h1
w6 += lr * delta_o1 * out_h2
w7 += lr * delta_o2 * out_h1
w8 += lr * delta_o2 * out_h2

w1 += lr * delta_h1 * i1
w2 += lr * delta_h1 * i2
w3 += lr * delta_h2 * i1
w4 += lr * delta_h2 * i2

b1 += lr * (delta_h1 + delta_h2)  
b2 += lr * (delta_o1 + delta_o2)  

print("Updated weights and biases:")
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")
print(f"w5: {w5}, w6: {w6}, w7: {w7}, w8: {w8}")
print(f"b1: {b1}, b2: {b2}")

print(f"Output after backpropagation: o1 = {out_o1}, o2 = {out_o2}")
