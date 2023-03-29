

module gp (g,p,c_in,g_out,p_out,c_out); // carry generator, carry propagator
    input [1:0] g, p;                       // lower  level 2-set of g, p
    input       c_in;                       // lower  level carry_in
    output      g_out,p_out,c_out;          // higher level g, p, carry_out
    assign      g_out = g[1] | p[1] & g[0]; // higher level carry generator
    assign      p_out = p[1] & p[0];        // higher level carry propagator
    assign      c_out = g[0] | p[0] & c_in; // higher level carry_out
endmodule


module add (a, b, c, g, p, s);                   // adder and g, p
    input  a, b, c;                              // inputs:  a, b, c;
    output g, p, s;                              // outputs: g, p, s; 
    assign s = a ^ b ^ c;                        // output: sum of inputs
    assign g = a & b;                            // output: carry generator
    assign p = a | b;                            // output: carry propagator
endmodule


module cla_2 (a, b, c_in, g_out, p_out, s);   // 2-bit carry lookahead adder
    input  [1:0] a, b;                                  // inputs:  a, b
    input        c_in;                                  // input:   carry_in
    output       g_out, p_out;                          // outputs: g, p
    output [1:0] s;                                     // output:  sum
    wire   [1:0] g, p;                                  // internal wires
    wire         c_out;                                 // internal wire
    // add (a,    b,    c,     g,    p,    s);          // generates g,p,s
    add a0 (a[0], b[0], c_in,  g[0], p[0], s[0]);       // add on bit 0
    add a1 (a[1], b[1], c_out, g[1], p[1], s[1]);       // add on bit 1
    // gp  (g, p, c_in, g_out, p_out, c_out);           // higher level g,p
    gp gp0 (g, p, c_in, g_out, p_out, c_out);           // higher level g,p
endmodule


module cla_4 (a,b,c_in,g_out,p_out,s);        // 4-bit carry lookahead adder
    input  [3:0] a, b;                                  // inputs:  a, b
    input        c_in;                                  // input:   carry_in
    output       g_out, p_out;                          // outputs: g, p
    output [3:0] s;                                     // output:  sum
    wire   [1:0] g, p;                                  // internal wires
    wire         c_out;                                 // internal wire
    cla_2 a0 (a[1:0],b[1:0],c_in, g[0],p[0],s[1:0]);    // add on bits 0,1
    cla_2 a1 (a[3:2],b[3:2],c_out,g[1],p[1],s[3:2]);    // add on bits 2,3
    gp   gp0 (g,p,c_in, g_out,p_out,c_out);             // higher level g,p
endmodule


module cla_8 (a,b,c_in,g_out,p_out,s);        // 8-bit carry lookahead adder
    input  [7:0] a, b;                                  // inputs:  a, b
    input        c_in;                                  // input:   carry_in
    output       g_out, p_out;                          // outputs: g, p
    output [7:0] s;                                     // output:  sum
    wire   [1:0] g, p;                                  // internal wires
    wire         c_out;                                 // internal wire
    cla_4 a0 (a[3:0],b[3:0],c_in, g[0],p[0],s[3:0]);    // add on bits 0-3
    cla_4 a1 (a[7:4],b[7:4],c_out,g[1],p[1],s[7:4]);    // add on bits 4-7
    gp   gp0 (g,p,c_in, g_out,p_out,c_out);             // higher level g,p
endmodule


module cla_16 (a,b,c_in,g_out,p_out,s);      // 16-bit carry lookahead adder
    input  [15:0] a, b;                                 // inputs:  a, b
    input         c_in;                                 // input:   carry_in
    output        g_out, p_out;                         // outputs: g, p
    output [15:0] s;                                    // output:  sum
    wire    [1:0] g, p;                                 // internal wires
    wire          c_out;                                // internal wire
    cla_8 a0 (a[7:0], b[7:0], c_in, g[0],p[0],s[7:0]);  // add on bits 0-7
    cla_8 a1 (a[15:8],b[15:8],c_out,g[1],p[1],s[15:8]); // add on bits 8-15
    gp   gp0 (g,p,c_in, g_out,p_out,c_out);             // higher level g,p
endmodule


module cla_32 (a,b,c_in,g_out,p_out,s);      // 32-bit carry lookahead adder
    input  [31:0] a, b;                                 // inputs:  a, b
    input         c_in;                                 // input:   carry_in
    output        g_out, p_out;                         // outputs: g, p
    output [31:0] s;                                    // output:  sum
    wire    [1:0] g, p;                                 // internal wires
    wire          c_out;                                // internal wire
    cla_16 a0 (a[15:0], b[15:0], c_in, g[0],p[0],s[15:0]);   // + bits 0-15
    cla_16 a1 (a[31:16],b[31:16],c_out,g[1],p[1],s[31:16]);  // + bits 16-31
    gp    gp0 (g,p,c_in,g_out,p_out,c_out);
endmodule


module cla32 (a,b,ci,s);    // 32-bit carry lookahead adder, no g, p outputs
    input  [31:0] a, b;                                 // inputs: a, b
    input         ci;                                   // input:  carry_in
    output [31:0] s;                                    // output: sum
    wire          g_out, p_out;                         // internal wires
    cla_32 cla (a, b, ci, g_out, p_out, s);             // use cla_32 module
endmodule


module addsub32 (a,b,sub,s);                      // 32-bit adder/subtracter
    input  [31:0] a, b;                           // inputs: a, b
    input         sub;                            // sub == 1: s = a - b
                                                  // sub == 0: s = a + b
    output [31:0] s;                              // output sum s
    // sub == 1: a - b = a + (-b) = a + not(b) + 1 = a + (b xor sub) + sub
    // sub == 0: a + b = a +   b  = a +     b  + 0 = a + (b xor sub) + sub
    wire   [31:0] b_xor_sub = b ^ {32{sub}};      // (b xor sub)
    // cla32   (a, b,         ci,  s);
    cla32 as32 (a, b_xor_sub, sub, s);            // b: (b xor sub); ci: sub
endmodule


module shift (d,sa,right,arith,sh);     // barrel shift, behavioral style
    input  [31:0] d;                    // input: 32-bit data to be shifted
    input   [4:0] sa;                   // input: shift amount, 5 bits
    input         right;                // 1: shift right; 0: shift left
    input         arith;                // 1: arithmetic shift; 0: logical
    output [31:0] sh;                   // output: shifted result
    reg    [31:0] sh;                   // will be combinational
    always @* begin                     // always block
        if (!right) begin               // if shift left
            sh = d << sa;               //    shift left sa bits
        end else if (!arith) begin      // if shift right logical
            sh = d >> sa;               //    shift right logical sa bits
        end else begin                  // if shift right arithmetic
            sh = $signed(d) >>> sa;     //    shift right arithmetic sa bits
        end
    end
endmodule


module mux4x32 (a0,a1,a2,a3,s,y); // 4-to-1 multiplexer, 32-bit
    input  [31:0] a0, a1, a2, a3; // inputs, 32 bits
    input   [1:0] s;              // input,   2 bits
    output [31:0] y;              // output, 32 bits
    function  [31:0] select;      // function name (= return value, 32 bits)
        input [31:0] a0,a1,a2,a3; // notice the order of the input arguments
        input  [1:0] s;           // notice the order of the input arguments
        case (s)                  // cases:
            2'b00: select = a0;   // if (s==0) return value = a0
            2'b01: select = a1;   // if (s==1) return value = a1
            2'b10: select = a2;   // if (s==2) return value = a2
            2'b11: select = a3;   // if (s==3) return value = a3
        endcase
    endfunction
    assign y = select(a0,a1,a2,a3,s);   // call the function with parameters
endmodule


module alu (a,b,aluc,r,z);           // 32-bit alu with a zero flag
    input  [31:0] a, b;              // inputs: a, b
    input   [3:0] aluc;              // input:  alu control: // aluc[3:0]:
    output [31:0] r;                 // output: alu result   // x 0 0 0  ADD
    output        z;                 // output: zero flag    // x 1 0 0  SUB
    wire   [31:0] d_and = a & b;                             // x 0 0 1  AND
    wire   [31:0] d_or  = a | b;                             // x 1 0 1  OR
    wire   [31:0] d_xor = a ^ b;                             // x 0 1 0  XOR
    wire   [31:0] d_lui = {b[15:0],16'h0};                   // x 1 1 0  LUI
    wire   [31:0] d_and_or  = aluc[2]? d_or  : d_and;        // 0 0 1 1  SLL
    wire   [31:0] d_xor_lui = aluc[2]? d_lui : d_xor;        // 0 1 1 1  SRL
    wire   [31:0] d_as, d_sh;                                // 1 1 1 1  SRA
    // addsub32   (a,b,sub,    s);
    addsub32 as32 (a,b,aluc[2],d_as);                        // add/sub
    // shift      (d,sa,    right,  arith,  sh);
    shift shifter (b,a[4:0],aluc[2],aluc[3],d_sh);           // shift
    // mux4x32  (a0,  a1,      a2,       a3,  s,        y);
    mux4x32 res (d_as,d_and_or,d_xor_lui,d_sh,aluc[1:0],r);  // alu result
    assign z = ~|r;                                          // z = (r == 0)
endmodule


module sccu_dataflow (op,func,z,wmem,wreg,regrt,m2reg,aluc,shift,aluimm,
                      pcsrc,jal,sext);            // control unit
  input  [5:0] op, func;                          // op, func 
  input        z;                                 // alu zero tag
  output [3:0] aluc;                              // alu operation control
  output [1:0] pcsrc;                             // select pc source
  output       wreg;                              // write regfile
  output       regrt;                             // dest reg number is rt
  output       m2reg;                             // instruction is an lw
  output       shift;                             // instruction is a shift
  output       aluimm;                            // alu input b is an i32
  output       jal;                               // instruction is a jal
  output       sext;                              // is sign extension
  output       wmem;                              // write data memory
  // decode instructions
  wire rtype  = ~|op;                                            // r format
  wire i_add  = rtype& func[5]&~func[4]&~func[3]&~func[2]&~func[1]&~func[0];
  wire i_sub  = rtype& func[5]&~func[4]&~func[3]&~func[2]& func[1]&~func[0];
  wire i_and  = rtype& func[5]&~func[4]&~func[3]& func[2]&~func[1]&~func[0];
  wire i_or   = rtype& func[5]&~func[4]&~func[3]& func[2]&~func[1]& func[0];
  wire i_xor  = rtype& func[5]&~func[4]&~func[3]& func[2]& func[1]&~func[0];
  wire i_sll  = rtype&~func[5]&~func[4]&~func[3]&~func[2]&~func[1]&~func[0];
  wire i_srl  = rtype&~func[5]&~func[4]&~func[3]&~func[2]& func[1]&~func[0];
  wire i_sra  = rtype&~func[5]&~func[4]&~func[3]&~func[2]& func[1]& func[0];
  wire i_jr   = rtype&~func[5]&~func[4]& func[3]&~func[2]&~func[1]&~func[0];
  wire i_addi = ~op[5]&~op[4]& op[3]&~op[2]&~op[1]&~op[0];       // i format
  wire i_andi = ~op[5]&~op[4]& op[3]& op[2]&~op[1]&~op[0];
  wire i_ori  = ~op[5]&~op[4]& op[3]& op[2]&~op[1]& op[0];
  wire i_xori = ~op[5]&~op[4]& op[3]& op[2]& op[1]&~op[0];
  wire i_lw   =  op[5]&~op[4]&~op[3]&~op[2]& op[1]& op[0];
  wire i_sw   =  op[5]&~op[4]& op[3]&~op[2]& op[1]& op[0];
  wire i_beq  = ~op[5]&~op[4]&~op[3]& op[2]&~op[1]&~op[0];
  wire i_bne  = ~op[5]&~op[4]&~op[3]& op[2]&~op[1]& op[0];
  wire i_lui  = ~op[5]&~op[4]& op[3]& op[2]& op[1]& op[0];
  wire i_j    = ~op[5]&~op[4]&~op[3]&~op[2]& op[1]&~op[0];       // j format
  wire i_jal  = ~op[5]&~op[4]&~op[3]&~op[2]& op[1]& op[0];
  // generate control signals
  assign regrt   = i_addi | i_andi | i_ori  | i_xori | i_lw  | i_lui;
  assign jal     = i_jal;
  assign m2reg   = i_lw;
  assign wmem    = i_sw;
  assign aluc[3] = i_sra;                         // refer to alu.v for aluc
  assign aluc[2] = i_sub  | i_or   | i_srl  | i_sra  | i_ori  | i_lui;
  assign aluc[1] = i_xor  | i_sll  | i_srl  | i_sra  | i_xori | i_beq |
                   i_bne  | i_lui;
  assign aluc[0] = i_and  | i_or | i_sll | i_srl | i_sra | i_andi | i_ori;
  assign shift   = i_sll  | i_srl  | i_sra;
  assign aluimm  = i_addi | i_andi | i_ori  | i_xori | i_lw  | i_lui | i_sw;
  assign sext    = i_addi | i_lw   | i_sw   | i_beq  | i_bne;
  assign pcsrc[1]= i_jr   | i_j    | i_jal;
  assign pcsrc[0]= i_beq & z | i_bne & ~z | i_j | i_jal;
  assign wreg    = i_add  | i_sub  | i_and  | i_or   | i_xor | i_sll  |
                   i_srl  | i_sra  | i_addi | i_andi | i_ori | i_xori |
                   i_lw   | i_lui  | i_jal;
endmodule


// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------


module regfile(input  logic        clk, 
               input  logic        we3, 
               input  logic [4:0]  ra1, ra2, wa3, 
               input  logic [31:0] wd3, 
               output logic [31:0] rd1, rd2);

  logic [31:0] rf[1:31];

  always_ff @(posedge clk)
    if (we3 && (we3 != 0)) rf[wa3] <= wd3;	

  assign rd1 = (ra1 == 0) ? 0 : rf[ra1];
  assign rd2 = (ra2 == 0) ? 0 : rf[ra2];
endmodule


module adder(input  logic [31:0] a, b,
             output logic [31:0] y);

  assign y = a + b;
endmodule


module flopr #(parameter WIDTH = 8)
              (input  logic             clk, reset,
               input  logic [WIDTH-1:0] d, 
               output logic [WIDTH-1:0] q);

  always_ff @(posedge clk, posedge reset)
    if (reset) q <= 0;
    else       q <= d;
endmodule


module mux2 #(parameter WIDTH = 8)
             (input  logic [WIDTH-1:0] d0, d1, 
              input  logic             s, 
              output logic [WIDTH-1:0] y);

  assign y = s ? d1 : d0; 
endmodule


module dmem(input  logic        clk, we,
            input  logic [31:0] a, wd,
            output logic [31:0] rd);

  logic [31:0] RAM[63:0];

  assign rd = RAM[a[31:2]]; // word aligned

  always_ff @(posedge clk)
    if (we) RAM[a[31:2]] <= wd;
endmodule


module imem(input  logic [5:0] a,
            output logic [31:0] rd);

  logic [31:0] RAM[63:0];

  initial
      $readmemh("memfile.dat",RAM);

  assign rd = RAM[a]; // word aligned
endmodule


module test(
	input logic clk, reset
);
	// program counter
	logic [31:0] pc, npc, p4, pcbranch, pcjump, _imm_shifted;
	assign _imm_shifted = {e[29:0], 2'b00};
	assign pcjump = {p4[31:28], addr, 2'b00};
	adder _p4(.a(pc), .b(32'd4), .y(p4));
	adder _pcbranch(.a(p4), .b(_imm_shifted), .y(pcbranch)); // pc + 4 + (extended imm)
	mux4x32 _npc(.a0(p4), .a1(pcbranch), .a2(qa), .a3(pcjump), .s(pcsrc), .y(npc));
	flopr #(32) _pc(.clk(clk), .reset(reset), .d(npc), .q(pc));

	// inst mem
	logic [31:0] instr;
	logic [5:0] op = instr[31:26];
	logic [5:0] func = instr[5:0];
	logic [4:0] rs = instr[25:21];
	logic [4:0] rt = instr[20:16];
	logic [4:0] rd = instr[15:11];
	logic [15:0] imm = instr[15:0];
	logic [25:0] addr = instr[25:0];
	imem _imem(.a(pc[7:2]), .rd(instr));

	// rd rt 31
	logic [4:0] reg_dest; // result of mux: rt / rd
	logic [4:0] wn; // addr input to regfile write data 
	assign wn = reg_dest | {5{jal}}; // = 11111 (reg 31) if jal=1
	mux2 #(5) _reg_dest(.d0(rd), .d1(rt), .s(regrt), .y(reg_dest)); 

	// wd 
	logic [31:0] wd; // input to wd port of regfile 
	mux2 #(32) _wd(.d0(r), .d1(p4), .s(jal), .y(wd));

	// regfile 
	logic [31:0] qa, qb; // outputs of reg file 
	regfile _regfile(.clk(clk), .we3(wreg), .ra1(rs), .ra2(rt), .wa3(wn), .wd3(wd), .rd1(qa), .rd2(qb));

	// extend imm 
	logic [31:0] e;
	assign e = {{16{sext & imm[15]}}, imm};

	// alu
	logic z;
	logic [31:0] aluresult, alua, alub;
	mux2 #(32) _alua(.d0(qa), .d1({27'b0, instr[10:6]}), .s(shift), .y(alua));
	mux2 #(32) _alub(.d0(qb), .d1(e), .s(aluimm), .y(alub));
	alu _alu(.a(alua), .b(alub), .aluc(aluc), .r(aluresult), .z(z));

	// data mem 
	logic [31:0] data, r;
	dmem _dmem(.clk(clk), .we(wmem), .a(aluresult), .wd(qb), .rd(data));
	mux2 #(32) _r(.d0(aluresult), .d1(data), .s(m2reg), .y(r));


	// control unit (which i copy-pasted)
	logic regrt, jal, sext, wreg, aluimm, shift, wmem, m2reg;
	logic [1:0] pcsrc;
	logic [3:0] aluc;
	sccu_dataflow _sccu_dataflow(.op(op), .func(func), .z(z), 
		.wmem(wmem), .wreg(wreg), .regrt(regrt), .m2reg(m2reg), .aluc(aluc), .shift(shift), .aluimm(aluimm),
		.pcsrc(pcsrc), .jal(jal), .sext(sext));

endmodule

