# generate test cases file


#	input logic cin,
#	input logic [7:0] a, b,
#	output logic [7:0] s,
#	output logic cout);


tests = [(cin, a, b) for cin in range(2) for a in range(256) for b in range(256)]

len(tests) # 131072

f = open('vectors.txt', 'w')

for test in tests:
    cin, a, b = test 

    s = (cin + a + b) % 256
    cout = ((cin + a + b) // 256) % 2

    _cin = bin(cin)[2:]
    _a = bin(a)[2:].rjust(8, '0')
    _b = bin(b)[2:].rjust(8, '0')
    _s = bin(s)[2:].rjust(8, '0')
    _cout = bin(cout)[2:]
    
    _line = f'{_cin}_{_a}_{_b}_{_s}_{_cout}\n'
    _ = f.write(_line)

f.close()
    
