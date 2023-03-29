# generate test cases file




tests = [(cin, a, b) for cin in range(2) for a in range(2) for b in range(2)]

f = open('vectors.txt', 'w')

for test in tests:
    cin, a, b = test 

    s = (cin + a + b) % 2
    cout = (cin + a + b) // 2 
    p = a | b 
    g = a & b 
    
    _line = f'{cin}_{a}_{b}_{s}_{cout}_{p}_{g}\n'
    _ = f.write(_line)

f.close()
    
