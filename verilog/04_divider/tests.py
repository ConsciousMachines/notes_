# generate test cases file


tests = [(a, b) for a in range(256) for b in range(1,256)]

f = open('vectors.txt', 'w')

for test in tests:
    a, b = test 

    mod = a % b 
    div = a // b 

    _a = bin(a)[2:].rjust(8, '0')
    _b = bin(b)[2:].rjust(8, '0')
    _mod = bin(mod)[2:].rjust(8, '0')
    _div = bin(div)[2:].rjust(8, '0')
    
    _line = f'{_a}_{_b}_{_mod}_{_div}\n'
    _ = f.write(_line)

f.close()

