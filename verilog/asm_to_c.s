


# logic R-type
and		$t0, $t1, $t2		# $t0 = $t1 & $t2
or		$s2, $t1, $t2		# $s2 = $t1 | $t2
xor		$t0, $t1, $t2		# $t0 = $t1 ^ $t2
nor		$t0, $t1, $t2		# $t0 = ~($t1 | $t2)
# A NOR $0 = NOT A 


# logic I-type
andi	$t0, $t1, 0			# $t0 = $t1 & 0
ori		$t0, $t1, 0			# $t0 = $t1 | 0
xori	$t0, $t1, 0			# $t0 = $t1 ^ 0
xori	$t0, $t1, 0xFAFA			# $t0 = $t1 ^ 0xFAFA

# shift (R-type but includes immediate value in instr)
sll		$t0, $t1, 0			# $t0 = $t1 << 0
srl		$t0, $t1, 0			# $t0 = $t1 >> 0
sra     $t1, $t2, 0x2
sllv $t0, $t1, $t2
srlv $t0, $t1, $t2
srav $t0, $t1, $t2

# generate 16-bit constant
addi $s0, $0, 0x1515 

# generate 32-bit constant
lui $s0, 0x1515 
ori $s0, $s0, 0x2525

# mult/divide: result placed in special registers hi & lo
mult $s0, $s1 
div $s0, $s1 
mul $s1, $s2, $s3
mfhi $s2 # copy result from special hi/lo registers
mflo $s1 # they are part of architectural state but we ignore them in this book 

# branch
beq $t0, $t1, target 
bne $t0, $t1, target 
target:
j target 
jr $s0 
jal 

# compare less than
slt $t1, $s0, $t0 # t1 = (s0 < t0) ? 1 : 0 

# conditionals
# if (s2 == s3)
#     stuff
# more stuff 
bne $s2, $s3, more_stuff # the conditional has been negated 
# stuff 
more_stuff:
# ... 
# ... 
# ... 
# if (s2 == s3)
#     stuff 
# else 
#     more_stuff
# finished
bne $s2, $s3, else 
# stuff 
j finished
else:
# more stuff
finished:
# ... 
# ... 
# ... 

# loops
# int s0 = 1; s1 = 0;
# while (s0 != 128) {
#     s0 = s0 * 2; s1 = s1 + 1; }
addi $s0, $0, 1 # pow = 1
addi $t0, $0, 120 # t0 = 128 for comparison
while: 
beq $s0, $t0, done # if pow == 128, exit while loop
sll $s0, $s0, 1 # pow = pow * 2 
j while 
done: # also do/while loops just execute the check at the end of the body
# ...
# ...
# ...
# s1 = 0
# for (s0 = 0; s0 != 10; s0 = s0 + 1) 
#     s1 = s1 + s0;
add $s1, $0, $0 # sum = 0
addi $s0, $0, 0 # i = 0
addi $t0, $0, 10 # t0 = 10 for comparison
for: 
beq $s0, $t0, done # if i == 10 goto done 
add $s1, $s1, $s0 # sum = sum + i 
addi $s0, $s0, 1 # i = i + 1
j for 
done: # literally a while loop with initialization code before loop + increment code after iteration
# ...
# ...
# ...
# s1 = 0 
# for (s0 = 1; s0 < 101; s0 = s0 * 2)
#     s1 = s1 + s0
addi $s1, $0, 0 # sum = 0
addi $s0, $0, 1 # i = 1
addi $t0, $0, 101 # t0 = 101 for comparison
loop:
slt $t1, $s0, $t0 # t1 = (s0 < t0) ? 1 : 0
beq $t1, $0, done # if (t1 == 0) goto done
add $s1, $s1, $s0 # sum = sum + i
sll $s0, $s0, 1 # i = i * 2
j loop 
done:

# accessing arrays
lui $s0, 0x1000 
ori $s0, $s0, 0x7000 # s0 = 0x10007000 base address of array
lw $t1, 4($s0) # t1 = *(s0 + 4)     # array[1] since byte-addr mem  
sw $t1, 4($s0) # *(s0 + 4) = t1

# byte memory
lbu $s1, 2($0) 
lb $s1, 2($0) 
sb $s1, 2($0) 

# character arrays
# char chararray[10];
# int i;
# for (i=0; i != 10; i = i + 1)
#     chararray[i] = chararray[i] - 32;
# s0 = base addr of chararray, s1 = i
addi $s1, $0, 0 # i = 0
addi $t0, $0, 10 # t0 = 10 for comparison
loop:
beq $t0, $s1, done # if i == 10 goto done
add $t1, $s1, $s0 # t1 = (array_base + offset_i)     # remember byte addressed/ &chararray[i]
lb $t2, 0($t1) # t2 = array[i]
addi $t2, $t2, -32 # convert to upper case: t2 -= 32
sb $t2 0($t1) # store new value in array: chararray[i] = t2
addi $s1, $s1, 1 # i = i + 1
j loop
done: # btw the arm book discusses using ptr arithmetic vs array access

# functions
# caller places up to four arguments in a0-a3 before calling the callee. 
# caller stores return address in $ra as it jumps to callee (jal)
# callee places return value in v0-v1
# callee must leave s0-s7, $ra, and stack unmodified 
# ...
# ...
# ...
# simple();
# void simple() {return;}
jal simple 
simple:
jr $ra 
# ...
# ...
# ...
# jal calls a function (like CALL in other asms)
# jr is used to return from fn (like RET in other asms)
# ...
# ...
# ...
# saving registers
# s0 is result
diffofsums:
addi $sp, $sp, -12 # make space on stack for 3 regs
sw $s0, 8($sp) # save s0 on stack
sw $t0, 4($sp) # save t0 on stack
sw $t1, 0($sp) # save t1 on stack
add $t0, $a0, $a1 # t0 = f + g
add $t1, $a2, $a3 # t1 = h + f
sub $s0, $t0, $t1 # result = (f + g) - (h + f)
add $v0, $s0, $0 # put return value in v0
lw $t1, 0($sp) # restore t1 from stack
lw $t0, 4($sp) # restore t0 from stack
lw $s0, 8($sp) # restore s0 from stack
addi $sp, $sp, 12 # deallocate stack space 
jr $ra # return to caller 
# ...
# ...
# ...
# stack frame: stack space the function allocates for itslelf. 
# preserved registers include: s0-s7 (must be saved)
# non-preserved registers include: t0-t9
# caller needs to save any non-preserved regs it wishes to keep
# ...
# ...
# ...
# in the factorial example, we can put things on the stack only in the ELSE case, where a fn call happens.
# we know to store a0 (n) and ra on the stack because they are non-preserved, and a fn call happens, and we need n (and ra, obv) later.
# ...
# ...
# ...
# in case a fn has >4 arguments, the caller expands its stack to place args there. 
#     the frame pointer is the SP after the extra args are passed, and before the fn call
#     it shows the start of the fn's stack frame, and additional args are accessed relative to FP since SP moves during fn call. 
#     since caller increments SP by # of additional args, after fn call it decrements stack likewise.
# a fn's local variables are usually in s0-s7, additional locals and arrays go on the stack. 
# TECHNICALLY for ease I can forgo the a0-a3 input and s0-s7 locals convention and put everything on the stack (for a first compiler)
#     in which case all registers are used for temporary results (still need to keep track of which (non)/preserved regs im using)

# addressing modes
# read/write operands:
#     register-only: use regs for source & destination of operands. ex: all R-type inst
#     immediate: use regs and 16-bit immediate field 
#     base: memory access like lw,sw: base address is in a register and offset is in imm field (so ptr ops)
# writing PC:
#     PC-relative: add imm field to PC
#         imm field is target_addr - (PC+4) (is it really though? what about the branch delay slot)
#         the processor sign-extends imm, << 2, add to PC+4
#     psuedo-direct: use 4 MSB of PC+4 and 28 bits from inst imm field << 2

# linker
# combines object files so they are not all on top of each other
# relocates the data/text sections. uses symbol tables to relocate the globals & labels

# coprocessor 0 - MIPS processor control, handles interrupts and processor diagnostics?
syscall 
break 
# these cause traps (software interrupts) to perform system calls and debugger breakpoints. 

# unsigned instructions
# in MIPS the default is signed, unsigned instr have 'u' on the end
addiu # subu addu # signed versions trigger exception on overflow, unsigned do not. C ignores overflow so it uses unsigned always.
multu 
divu 
sltu 
sltiu 

# NOTE: in Yamin, we are not allowed to put syscall in a delay slot.
# similarly, no unimplemented instr in a delay slot.



# MIPS ASSEMBLY LANGUAGE PROGRAMMING BOOK 

# $at, the assembler temporary register, is used by the assembler to implement macro instructions. 

# * * * BASED example of switch statement in MIPS assembly book page 16 uses jump table

# allocate space
.data # all following data allocation directives should allocate data in the data segment.  
ARRAY: .space 4096

Pof2: .word 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096

# access element two in the array and put a copy in $s0
la $a0 Pof2 # a0 = &Pof2 
lw $s0 8($a0) # s0 = MEM[a0 + 8]
# load address (la) macro puts pointer to Pof2 in a0


Buffer: .space 60
# system call to read string 
li $v0 8 # syscall to read string 
la $a0 Bufer # ptr to buffer to store string 
li $a1 60 # max length of input buffer
syscall 


.data 
prompt: .asciiz "\n bruh please write a number"
result: .asciiz "the finkel swagoose"
bye: .asciiz "watashi wa soy boy desu ga"
soy_array: .word 420, 69
.globl main 
.text 
main: addi $t0 $0 0x1 

# use addiu for sp so we dont generate exception when we cross over midpoint of mem
addiu $sp $sp -24


