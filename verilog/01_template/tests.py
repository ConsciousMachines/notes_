# generate test cases file

tests = [(i, j) for i in range(2) for j in range(2)]

f = open('vectors.txt', 'w')

for test in tests:
    i, j = test
    _and = i & j
    _line = f'{i}_{j}_{_and}\n'
    _ = f.write(_line)

f.close()
    
