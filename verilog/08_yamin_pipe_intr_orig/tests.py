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