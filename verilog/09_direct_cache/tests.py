# generate test cases file


soy = '''
mwreg,mrn,ern,ewreg,em2reg,mm2reg,rsrtequ,func,op,rs,rt,
                   rd,op1,wreg,m2reg,wmem,aluc,regrt,aluimm,fwda,fwdb,wpcir,
                   sext,pcsrc,shift,jal,irq,sta,ecancel,eis_branch,
                   mis_branch,inta,selpc,exc,sepc,cause,mtc0,wepc,wcau,wsta,
                   mfc0,is_branch,ove,cancel,exc_ovr,mexc_ovr
'''.split(',')
soy = [i.strip() for i in soy]
soy 
for i in soy:
    print(f'.{i}({i}),')
