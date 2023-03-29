# generate test cases file


soy = '''
mwreg,mrn,ern,ewreg,em2reg,mm2reg,rsrtequ,func,op,rs,rt,
                 wreg,m2reg,wmem,aluc,regrt,aluimm,fwda,fwdb,nostall,sext,
                 pcsrc,shift,jal
'''.split(',')
soy = [i.strip() for i in soy]
soy 
for i in soy:
    print(f'.{i}({i}),')




f1 = open('wave_my.vcd', 'r')
f2 = open('wave_orig.vcd', 'r')

l1 = f1.readlines()
l2 = f2.readlines()

num = 0
for i, j in zip(l1, l2):
    num += 1
    if i != j:
        print(i, j, num)
        break