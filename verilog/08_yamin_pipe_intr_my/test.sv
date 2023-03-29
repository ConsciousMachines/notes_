









module pipelined_cpu_exc_int (clk,clrn,pc,inst,ealu,malu,wdi,intr,inta);
    input         clk, clrn;               // clock and reset
    input         intr;                    // interrupt request
    output [31:0] pc;                      // program counter
    output [31:0] inst;                    // instruction in ID stage
    output [31:0] ealu;                    // result in EXE stage
    output [31:0] malu;                    // result in MEM stage
    output [31:0] wdi;                     // result in WB stage
    output        inta;                    // interrupt acknowledgement
    parameter     exc_base = 32'h00000008; // exception handler entry
    // signals in IF stage
    wire   [31:0] pc4,ins,npc;
    wire   [31:0] next_pc;                 // next pc
    // signals in ID stage
    wire   [31:0] dpc4,bpc,jpc,da,db,imm,qa,qb;
    wire    [5:0] op,func;
    wire    [4:0] rs,rt,rd,rn;
    wire    [3:0] aluc;
    wire    [1:0] pcsrc,fwda,fwdb;
    wire          wreg,m2reg,wmem,aluimm,shift,jal,sext,regrt,rsrtequ,wpcir;
    wire   [31:0] pcd;                     // pc in ID stage
    wire   [31:0] cause;                   // cause content
    wire   [31:0] sta_in;                  // status register, data in
    wire   [31:0] cau_in;                  // cause  register, data in
    wire   [31:0] epc_in;                  // epc    register, data in
    wire   [31:0] epcin;                   // pc, pcd, pce, or pcm
    wire   [31:0] stalr;                   // state shift left or right
    wire    [1:0] mfc0;                    // select pc+8, sta, cau, or epc
    wire    [1:0] selpc;                   // select for next_pc
    wire    [1:0] sepc;                    // select for epcin
    wire          isbr;                    // is a branch or a jump
    wire          ove;                     // ov enable = arith & sta[3]
    wire          cancel;                  // cancel next instruction
    wire          exc;                     // exc or int occurs
    wire          mtc0;                    // move to c0 instruction
    wire          wsta;                    // status register write enable
    wire          wcau;                    // cause  register write enable
    wire          wepc;                    // epc    register write enable
    wire          irq;                     // latched intr
    // signals in EXE stage
    wire   [31:0] ealua,ealub,esa,ealu0,epc8;
    reg    [31:0] ea,eb,eimm,epc4;
    reg     [4:0] ern0;
    wire    [4:0] ern;
    reg     [3:0] ealuc;
    reg           ewreg0,em2reg,ewmem,ealuimm,eshift,ejal;
    wire          ewreg,zero;
    wire          exc_ovr;                 // overflow exc in EXE stage
    reg    [31:0] pce;                     // pc in EXE stage
    wire   [31:0] sta;                     // status register, data out
    wire   [31:0] cau;                     // cause  register, data out
    wire   [31:0] epc;                     // epc    register, data out
    wire   [31:0] pc8c0r;                  // epc8, sta, cau, or epc
    reg     [1:0] emfc0;                   // mfc0   in EXE stage
    reg           eisbr;                   // isbr   in EXE stage
    reg           eove;                    // ove    in EXE stage
    reg           ecancel;                 // cancel in EXE stage
    wire          ov;                      // overflow flag
    // signals in MEM stage
    wire   [31:0] mmo;
    reg    [31:0] malu,mb;
    reg     [4:0] mrn;
    reg           mwreg,mm2reg,mwmem;
    reg    [31:0] pcm;                     // pc      in MEM stage
    reg           misbr;                   // isbr    in MEM stage
    reg           mexc_ovr;                // exc_ovr in MEM stage
    // signals in WB stage
    reg    [31:0] wmo,walu;
    reg     [4:0] wrn;
    reg           wwreg,wm2reg;
    // program counter
    dffe32 prog_cnt (next_pc,clk,clrn,wpcir,pc);      // pc
    // IF stage
    cla32 pc_plus4 (pc,32'h4,1'b0,pc4);               // pc+4
    mux4x32 nextpc (pc4,bpc,da,jpc,pcsrc,npc);        // next pc
    pl_exc_i_mem inst_mem (pc,ins);                   // inst mem
    // IF/ID pipeline register
    dffe32 pc_4_r (pc4,clk,clrn,wpcir,dpc4);          // pc+4 reg
    dffe32 inst_r (ins,clk,clrn,wpcir,inst);          // ir
    dffe32 pcd_r  ( pc,clk,clrn,wpcir,pcd);           // pcd reg
    dff    intr_r (intr,clk,clrn,irq);                // interrupt req reg
    // ID stage
    assign op   = inst[31:26];                        // op
    assign rs   = inst[25:21];                        // rs
    assign rt   = inst[20:16];                        // rt
    assign rd   = inst[15:11];                        // rd
    assign func = inst[05:00];                        // func
    assign imm  = {{16{sext&inst[15]}},inst[15:0]};
    assign jpc  = {dpc4[31:28],inst[25:0],2'b00};     // jump target
    regfile rf (rs,rt,wdi,wrn,wwreg,~clk,clrn,qa,qb); // reg file
    mux2x5  des_reg_no (rd,rt,regrt,rn);              // destination reg
    mux4x32 operand_a (qa,ealu,malu,mmo,fwda,da);     // forward a
    mux4x32 operand_b (qb,ealu,malu,mmo,fwdb,db);     // forward b
    assign  rsrtequ = ~|(da^db);                      // rsrtequ = (da==db)
    cla32 br_addr (dpc4,{imm[29:0],2'b00},1'b0,bpc);  // branch target
    cu_exc_int cu (mwreg,mrn,ern,ewreg,em2reg,mm2reg,rsrtequ,func,op,rs,
                   rt,rd,rs,wreg,m2reg,wmem,aluc,regrt,aluimm,fwda,fwdb,
                   wpcir,sext,pcsrc,shift,jal,irq,sta,ecancel,eisbr,misbr,
                   inta,selpc,exc,sepc,cause,mtc0,wepc,wcau,wsta,mfc0,isbr,
                   ove,cancel,exc_ovr,mexc_ovr);
    dffe32  c0_status (sta_in,clk,clrn,wsta,sta);     // status register
    dffe32  c0_cause  (cau_in,clk,clrn,wcau,cau);     // cause register
    dffe32  c0_epc    (epc_in,clk,clrn,wepc,epc);     // epc register
    mux2x32 sta_mx (stalr,db,mtc0,sta_in);            // mux for status reg
    mux2x32 cau_mx (cause,db,mtc0,cau_in);            // mux for cause reg
    mux2x32 epc_mx (epcin,db,mtc0,epc_in);            // mux for epc reg
    mux2x32 sta_lr ({4'h0,sta[31:4]},{sta[27:0],4'h0},exc,stalr);
    mux4x32 epc_10 (pc,pcd,pce,pcm,sepc,epcin);       // select epc source
    mux4x32 irq_pc (npc,epc,exc_base,32'h0,selpc,next_pc); // for pc
    mux4x32 fromc0 (epc8,sta,cau,epc,emfc0,pc8c0r);   // for mfc0
    // ID/EXE pipeline register
    always @(negedge clrn or posedge clk)
      if (!clrn) begin
          ewreg0 <= 0;          em2reg  <= 0;          ewmem <= 0;
          ealuc  <= 0;          ealuimm <= 0;          ea    <= 0;
          eb     <= 0;          eimm    <= 0;          ern0  <= 0;
          eshift <= 0;          ejal    <= 0;          epc4  <= 0;
          eove   <= 0;          ecancel <= 0;          eisbr <= 0;
          emfc0  <= 0;          pce     <= 0;
      end else begin
          ewreg0 <= wreg;       em2reg  <= m2reg;      ewmem <= wmem;
          ealuc  <= aluc;       ealuimm <= aluimm;     ea    <= da;
          eb     <= db;         eimm    <= imm;        ern0  <= rn;
          eshift <= shift;      ejal    <= jal;        epc4  <= dpc4;
          eove   <= ove;        ecancel <= cancel;     eisbr <= isbr;
          emfc0  <= mfc0;       pce     <= pcd;
      end
    // EXE stage
    assign      esa = {eimm[5:0],eimm[31:6]};
    cla32   ret_addr (epc4,32'h4,1'b0,epc8);
    mux2x32 alu_ina  (ea,esa,eshift,ealua);
    mux2x32 alu_inb  (eb,eimm,ealuimm,ealub);
    mux2x32 save_pc8 (ealu0,pc8c0r,ejal|emfc0[1]|emfc0[0],ealu); // c0 regs
    assign  ern = ern0 | {5{ejal}};
    alu_ov  al_unit (ealua,ealub,ealuc,ealu0,zero,ov);
    assign  exc_ovr = ov & eove;                     // overflow exception
    assign  ewreg   = ewreg0 & ~exc_ovr;             // cancel overflow inst
    // EXE/MEM pipeline register
    always @(negedge clrn or posedge clk)
      if (!clrn) begin
          mwreg  <= 0;          mm2reg  <= 0;          mwmem <= 0;
          malu   <= 0;          mb      <= 0;          mrn   <= 0;
          misbr  <= 0;          pcm     <= 0;          mexc_ovr <= 0;
      end else begin
          mwreg  <= ewreg;      mm2reg  <= em2reg;     mwmem <= ewmem;
          malu   <= ealu;       mb      <= eb;         mrn   <= ern;
          misbr  <= eisbr;      pcm     <= pce;        mexc_ovr <= exc_ovr;
      end
    // MEM stage
    pl_exc_d_mem data_mem (clk,mmo,mb,malu,mwmem);   // data mem
    // MEM/WB pipeline register
    always @(negedge clrn or posedge clk)
      if (!clrn) begin
          wwreg  <= 0;          wm2reg  <= 0;          wmo   <= 0;
          walu   <= 0;          wrn     <= 0;
      end else begin
          wwreg  <= mwreg;      wm2reg  <= mm2reg;     wmo   <= mmo;
          walu   <= malu;       wrn     <= mrn;
      end
    // WB stage
    mux2x32 wb_stage (walu,wmo,wm2reg,wdi);          // alu res or mem data
endmodule

module dff (d,clk,clrn,q);            // dff with asynchronous reset
    input      d, clk, clrn;          // inputs d, clk, clrn (active low)
    output reg q;                     // output q, register type
    always @ (posedge clk or negedge clrn) begin // always block, "or"
        if (!clrn) q <= 0;            // if clrn is asserted, reset dff
        else       q <= d;            // else store d to dff
    end
endmodule

module add (a, b, c, g, p, s);                   // adder and g, p
    input  a, b, c;                              // inputs:  a, b, c;
    output g, p, s;                              // outputs: g, p, s; 
    assign s = a ^ b ^ c;                        // output: sum of inputs
    assign g = a & b;                            // output: carry generator
    assign p = a | b;                            // output: carry propagator
endmodule

module gp (g,p,c_in,g_out,p_out,c_out); // carry generator, carry propagator
    input [1:0] g, p;                       // lower  level 2-set of g, p
    input       c_in;                       // lower  level carry_in
    output      g_out,p_out,c_out;          // higher level g, p, carry_out
    assign      g_out = g[1] | p[1] & g[0]; // higher level carry generator
    assign      p_out = p[1] & p[0];        // higher level carry propagator
    assign      c_out = g[0] | p[0] & c_in; // higher level carry_out
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

module alu_ov (a,b,aluc,r,z,v);   // 32-bit alu with zero and overflow flags
    input  [31:0] a, b;           // inputs:  a, b
    input   [3:0] aluc;           // input:   alu control:   // aluc[3:0]:
    output [31:0] r;              // output:  alu result     // x 0 0 0  ADD
    output        z, v;           // outputs: zero, overflow // x 1 0 0  SUB
    wire   [31:0] d_and = a & b;                             // x 0 0 1  AND
    wire   [31:0] d_or  = a | b;                             // x 1 0 1  OR
    wire   [31:0] d_xor = a ^ b;                             // x 0 1 0  XOR
    wire   [31:0] d_lui = {b[15:0],16'h0};                   // x 1 1 0  LUI
    wire   [31:0] d_and_or  = aluc[2]? d_or  : d_and;        // 0 0 1 1  SLL
    wire   [31:0] d_xor_lui = aluc[2]? d_lui : d_xor;        // 0 1 1 1  SRL
    wire   [31:0] d_as,d_sh;                                 // 1 1 1 1  SRA
    // addsub32   (a,b,sub,    s);
    addsub32 as32 (a,b,aluc[2],d_as);                        // add/sub
    // shift      (d,sa,    right,  arith,  sh);
    shift shifter (b,a[4:0],aluc[2],aluc[3],d_sh);           // shift
    // mux4x32  (a0,  a1,      a2,       a3,  s,        y);
    mux4x32 res (d_as,d_and_or,d_xor_lui,d_sh,aluc[1:0],r);  // alu result
    assign z = ~|r;                                          // z = (r == 0)
    assign v = ~aluc[2] & ~a[31] & ~b[31] &  r[31] & ~aluc[1] & ~aluc[0] |
               ~aluc[2] &  a[31] &  b[31] & ~r[31] & ~aluc[1] & ~aluc[0] |
                aluc[2] & ~a[31] &  b[31] &  r[31] & ~aluc[1] & ~aluc[0] |
                aluc[2] &  a[31] & ~b[31] & ~r[31] & ~aluc[1] & ~aluc[0];
endmodule

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

module dffe32 (d,clk,clrn,e,q);                         // a 32-bit register
    input      [31:0] d;                                // input d
    input             e;                                // e: enable
    input             clk, clrn;                        // clock and reset
    output reg [31:0] q;                                // output q
    always @(negedge clrn or posedge clk)
        if (!clrn)  q <= 0;                             // q = 0 if reset
        else if (e) q <= d;                             // save d if enabled
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

module mux2x32 (a0,a1,s,y);                    // multiplexer, 32 bits
    input  [31:0] a0, a1;                      // inputs, 32 bits
    input         s;                           // input,   1 bit
    output [31:0] y;                           // output, 32 bits
    assign        y = s ? a1 : a0;             // if (s==1) y=a1; else y=a0;
endmodule

module mux2x5 (a0,a1,s,y);
    input  [4:0] a0,a1;
    input        s;
    output [4:0] y;
    assign y = s? a1 : a0;
endmodule

module regfile (rna,rnb,d,wn,we,clk,clrn,qa,qb);     // 32x32 regfile
    input  [31:0] d;                                 // data of write port
    input   [4:0] rna;                               // reg # of read port A
    input   [4:0] rnb;                               // reg # of read port B
    input   [4:0] wn;                                // reg # of write port
    input         we;                                // write enable
    input         clk, clrn;                         // clock and reset
    output [31:0] qa, qb;                            // read ports A and B
    reg    [31:0] register [1:31];                   // 31 32-bit registers
    assign qa = (rna == 0)? 0 : register[rna];       // read port A
    assign qb = (rnb == 0)? 0 : register[rnb];       // read port B
    integer i;
    always @(posedge clk or negedge clrn)            // write port
        if (!clrn)
            for (i = 1; i < 32; i = i + 1)
                register[i]  <= 0;                   // reset
        else
            if ((wn != 0) && we)                     // not reg[0] & enabled
                register[wn] <= d;                   // write d to reg[wn]
endmodule

module pl_exc_d_mem (clk,dataout,datain,addr,we); // data memory, ram
    input         clk;                     // clock
    input         we;                      // write enable
    input  [31:0] datain;                  // data in (to memory)
    input  [31:0] addr;                    // ram address
    output [31:0] dataout;                 // data out (from memory)
    reg    [31:0] ram [0:31];              // ram cells: 32 words * 32 bits
    assign dataout = ram[addr[6:2]];       // use 6-bit word address
    always @ (posedge clk) begin
        if (we) ram[addr[6:2]] = datain;   // write ram
    end
    integer i;
    initial begin                          // ram initialization
        for (i = 0; i < 32; i = i + 1)
            ram[i] = 0;
        // ram[word_addr] = data           // (byte_addr) item in data array
        ram[5'h08] = 32'h00000030;         // (20) 0. int_entry
        ram[5'h09] = 32'h0000003c;         // (24) 1. sys_entry
        ram[5'h0a] = 32'h00000054;         // (28) 2. uni_entry
        ram[5'h0b] = 32'h00000068;         // (2c) 3. ovr_entry
        ram[5'h12] = 32'h00000002;         // (48) for testing overflow
        ram[5'h13] = 32'h7fffffff;         // (4c) 2 + max_int -> overflow
        ram[5'h14] = 32'h000000a3;         // (50) data[0]   0 +  a3 =  a3
        ram[5'h15] = 32'h00000027;         // (54) data[1]  a3 +  27 =  ca
        ram[5'h16] = 32'h00000079;         // (58) data[2]  ca +  79 = 143
        ram[5'h17] = 32'h00000115;         // (5c) data[3] 143 + 115 = 258
    end
endmodule

module pl_exc_i_mem (a,inst);         // instruction memory, rom
    input  [31:0] a;                  // address
    output [31:0] inst;               // instruction
    wire   [31:0] rom [0:63];         // rom cells: 64 words * 32 bits
    // rom[word_addr] = instruction   // (pc) label      instruction
    assign rom[6'h00] = 32'h0800001d; // (00) main:      j    start
    assign rom[6'h01] = 32'h00000000; // (04)            nop
    // common entry of exc and intr
    assign rom[6'h02] = 32'h401a6800; // (08) exc_base:  mfc0 $26, c0_cause
    assign rom[6'h03] = 32'h335b000c; // (0c)            andi $27, $26, 0xc
    assign rom[6'h04] = 32'h8f7b0020; // (10)            lw $27,j_table($27)
    assign rom[6'h05] = 32'h00000000; // (14)            nop
    assign rom[6'h06] = 32'h03600008; // (18)            jr   $27
    assign rom[6'h07] = 32'h00000000; // (1c)            nop
    // 0x00000030: intr handler
    assign rom[6'h0c] = 32'h00000000; // (30) int_entry: nop
    assign rom[6'h0d] = 32'h42000018; // (34)            eret
    assign rom[6'h0e] = 32'h00000000; // (38)            nop
    // 0x0000003c: syscall handler
    assign rom[6'h0f] = 32'h00000000; // (3c) sys_entry: nop
    assign rom[6'h10] = 32'h401a7000; // (40) epc_plus4: mfc0 $26, c0_epc
    assign rom[6'h11] = 32'h235a0004; // (44)            addi $26, $26, 4
    assign rom[6'h12] = 32'h409a7000; // (48)            mtc0 $26, c0_EPC
    assign rom[6'h13] = 32'h42000018; // (4c) e_return:  eret
    assign rom[6'h14] = 32'h00000000; // (50)            nop
    // 0x00000054: unimpl handler
    assign rom[6'h15] = 32'h00000000; // (54) uni_entry: nop
    assign rom[6'h16] = 32'h08000010; // (58)            j    epc_plus4
    assign rom[6'h17] = 32'h00000000; // (5c)            nop
    // 0x00000068: overflow handler
    assign rom[6'h1a] = 32'h00000000; // (68) ovf_entry: nop
    assign rom[6'h1b] = 32'h0800002f; // (6c)            j    exit
    assign rom[6'h1c] = 32'h00000000; // (70)            nop
    // start: enable exc and intr
    assign rom[6'h1d] = 32'h2008000f; // (74) start:     addi $8, $0, 0xf
    assign rom[6'h1e] = 32'h40886000; // (78) exc_ena:   mtc0 $8, c0_status
    // unimplemented instruction
    assign rom[6'h1f] = 32'h0128001a; // (7c) unimpl:    div  $9, $8
    assign rom[6'h20] = 32'h00000000; // (80)            nop
    // system call
    assign rom[6'h21] = 32'h0000000c; // (84) sys:       syscall
    assign rom[6'h22] = 32'h00000000; // (88)            nop
    // loop code for testing intr
    assign rom[6'h23] = 32'h34040050; // (8c) int:       ori  $4, $1, 0x50
    assign rom[6'h24] = 32'h20050004; // (90)            addi $5, $0, 4
    assign rom[6'h25] = 32'h00004020; // (94)            add  $8, $0, $0
    assign rom[6'h26] = 32'h8c890000; // (98) loop:      lw   $9, 0($4)
    assign rom[6'h27] = 32'h01094020; // (9c)            add  $8, $8, $9
    assign rom[6'h28] = 32'h20a5ffff; // (a0)            addi $5, $5, -1
    assign rom[6'h29] = 32'h14a0fffc; // (a4)            bne  $5, $0, loop
    assign rom[6'h2a] = 32'h20840004; // (a8)            addi $4, $4, 4 # DS
    assign rom[6'h2b] = 32'h8c080048; // (ac) ov:        lw   $8, 0x48($0)
    assign rom[6'h2c] = 32'h8c09004c; // (b0)            lw   $9, 0x4c($0)
    // jump to start forever
    assign rom[6'h2d] = 32'h0800001d; // (b4) forever:   j    start
    // overflow in delay slot
    assign rom[6'h2e] = 32'h01094020; // (b8)            add  $9, $9, $8 #ov
    // if not overflow, go to start
    // exit, should be jal $31 to os
    assign rom[6'h2f] = 32'h0800002f; // (bc) exit:      j    exit
    assign rom[6'h30] = 32'h00000000; // (c0)            nop
    assign inst = rom[a[7:2]];        // use 6-bit word address to read rom
endmodule

module cu_exc_int (mwreg,mrn,ern,ewreg,em2reg,mm2reg,rsrtequ,func,op,rs,rt,
                   rd,op1,wreg,m2reg,wmem,aluc,regrt,aluimm,fwda,fwdb,wpcir,
                   sext,pcsrc,shift,jal,irq,sta,ecancel,eis_branch,
                   mis_branch,inta,selpc,exc,sepc,cause,mtc0,wepc,wcau,wsta,
                   mfc0,is_branch,ove,cancel,exc_ovr,mexc_ovr); // ctrl unit
    input  [31:0] sta;                // status: IM[3:0]: ov,unimpl,sys,int 
    input   [5:0] op,func;
    input   [4:0] mrn,ern,rs,rt,rd;
    input   [4:0] op1;                // for decode mfc0, mtc0, and eret
    input         mwreg,ewreg,em2reg,mm2reg,rsrtequ;
    input         irq;                // interrupt request
    input         ecancel;            // cancel in EXE stage
    input         eis_branch;         // is_branch in EXE stage
    input         mis_branch;         // is_branch in MEM stage
    input         exc_ovr;            // overflow exception occurs
    input         mexc_ovr;           // exc_ovr in MEM stage
    output [31:0] cause;              // cause content
    output  [3:0] aluc;
    output  [1:0] pcsrc,fwda,fwdb;
    output  [1:0] selpc;              // 00: npc;  01: epc; 10: exc_base
    output  [1:0] mfc0;               // 00: epc8; 01: sta; 10: cau; 11: epc
    output  [1:0] sepc;               // 00: pc;   01: pcd; 10: pce; 11: pcm
    output        wpcir,wreg,m2reg,wmem,regrt,aluimm,sext,shift,jal;
    output        inta;               // interrupt acknowledgement
    output        exc;                // any int or exc happened
    output        mtc0;               // is mtc0 instruction
    output        wsta;               // status register write enable
    output        wcau;               // cause  register write enable
    output        wepc;               // epc    register write enable
    output        is_branch;          // is a branch or a jump
    output        ove;                // ov enable = arith & sta[3]
    output        cancel;             // exception cancels next instruction
    reg     [1:0] fwda,fwdb;
    wire    [1:0] exccode;            // exccode
    wire          rtype,i_add,i_sub,i_and,i_or,i_xor,i_sll,i_srl,i_sra;
    wire          i_jr,i_addi,i_andi,i_ori,i_xori,i_lw,i_sw,i_beq,i_bne;
    wire          i_lui,i_j,i_jal,i_rs,i_rt;
    wire          exc_int;            // exception of interrupt
    wire          exc_sys;            // exception of system call
    wire          exc_uni;            // exception of unimplemented inst
    wire          c0_type;            // cp0 instructions
    wire          i_syscall;          // is syscall instruction
    wire          i_mfc0;             // is mfc0 instruction
    wire          i_mtc0;             // is mtc0 instruction
    wire          i_eret;             // is eret instruction
    wire          unimplemented_inst; // is an unimplemented inst
    wire          rd_is_status;       // rd is status
    wire          rd_is_cause;        // rd is cause
    wire          rd_is_epc;          // rd is epc
    wire   arith     = i_add | i_sub | i_addi;             // for overflow
    assign is_branch = i_beq | i_bne | i_jr | i_j | i_jal; // has delay slot
    assign exc_int   = sta[0] & irq;                    // 0. exc_int
    assign exc_sys   = sta[1] & i_syscall;              // 1. exc_sys
    assign exc_uni   = sta[2] & unimplemented_inst;     // 2. exc_uni
    assign ove       = sta[3] & arith;                  // 3. exc_ovr enable
    assign inta      = exc_int;                         // ack immediately
    assign exc       = exc_int | exc_sys | exc_uni | exc_ovr; // all int_exc
    assign cancel    = exc | i_eret;   // always cancel next inst, eret also
    // sel epc:    id is_branch   eis_branch    mis_branch     others
    // exc_int     PCD (01)       PC  (00)      PC  (00)       PC  (00)
    // exc_sys     x              x             PCD (01)       PCD (01)
    // exc_uni     x              x             PCD (01)       PCD (01)
    // exc_ovr     x              x             PCM (11)       PCE (10)
    assign sepc[0] = exc_int &  is_branch | exc_sys | exc_uni |
                     exc_ovr & mis_branch;
    assign sepc[1] = exc_ovr;
    // exccode:  0 0 : irq
    //           0 1 : i_syscall
    //           1 0 : unimplemented_inst
    //           1 1 : exc_ovr
    assign exccode[0]   = i_syscall          | exc_ovr;
    assign exccode[1]   = unimplemented_inst | exc_ovr;
    assign cause        = {eis_branch,27'h0,exccode,2'b00}; // BD
    assign mtc0         = i_mtc0;
    assign wsta         = exc | mtc0 & rd_is_status | i_eret;
    assign wcau         = exc | mtc0 & rd_is_cause;
    assign wepc         = exc | mtc0 & rd_is_epc;
    assign rd_is_status = (rd == 5'd12);              // cp0 status register
    assign rd_is_cause  = (rd == 5'd13);              // cp0 cause register
    assign rd_is_epc    = (rd == 5'd14);              // cp0 epc register
    // mfc0:     0 0 : epc8
    //           0 1 : sta
    //           1 0 : cau
    //           1 1 : epc
    assign mfc0[0] = i_mfc0 & rd_is_status | i_mfc0 & rd_is_epc;
    assign mfc0[1] = i_mfc0 & rd_is_cause  | i_mfc0 & rd_is_epc;
    // selpc:    0 0 : npc
    //           0 1 : epc
    //           1 0 : exc_base
    //           1 1 : x
    assign selpc[0] = i_eret;
    assign selpc[1] = exc;
    assign c0_type  = ~op[5]  & op[4]  & ~op[3] & ~op[2] & ~op[1] & ~op[0];
    assign i_mfc0   = c0_type &~op1[4] &~op1[3] &~op1[2] &~op1[1] &~op1[0];
    assign i_mtc0   = c0_type &~op1[4] &~op1[3] & op1[2] &~op1[1] &~op1[0];
    assign i_eret   = c0_type & op1[4] &~op1[3] &~op1[2] &~op1[1] &~op1[0] &
                 ~func[5] & func[4] & func[3] &~func[2] &~func[1] &~func[0];
    assign i_syscall = rtype  & ~func[5] & ~func[4] & func[3] & func[2] &
                      ~func[1] & ~func[0];
    assign unimplemented_inst = ~(i_mfc0 | i_mtc0 | i_eret | i_syscall |
           i_add | i_sub  | i_and  | i_or | i_xor | i_sll | i_srl | i_sra |
           i_jr  | i_addi | i_andi | i_ori | i_xori | i_lw | i_sw | i_beq |
           i_bne | i_lui  | i_j    | i_jal); // except for implemented insts
    and (rtype,~op[5],~op[4],~op[3],~op[2],~op[1],~op[0]);       // r format
    and (i_add,rtype, func[5],~func[4],~func[3],~func[2],~func[1],~func[0]);
    and (i_sub,rtype, func[5],~func[4],~func[3],~func[2], func[1],~func[0]);
    and (i_and,rtype, func[5],~func[4],~func[3], func[2],~func[1],~func[0]);
    and (i_or, rtype, func[5],~func[4],~func[3], func[2],~func[1], func[0]);
    and (i_xor,rtype, func[5],~func[4],~func[3], func[2], func[1],~func[0]);
    and (i_sll,rtype,~func[5],~func[4],~func[3],~func[2],~func[1],~func[0]);
    and (i_srl,rtype,~func[5],~func[4],~func[3],~func[2], func[1],~func[0]);
    and (i_sra,rtype,~func[5],~func[4],~func[3],~func[2], func[1], func[0]);
    and (i_jr, rtype,~func[5],~func[4], func[3],~func[2],~func[1],~func[0]);
    and (i_addi,~op[5],~op[4], op[3],~op[2],~op[1],~op[0]);      // i format
    and (i_andi,~op[5],~op[4], op[3], op[2],~op[1],~op[0]);
    and (i_ori, ~op[5],~op[4], op[3], op[2],~op[1], op[0]);
    and (i_xori,~op[5],~op[4], op[3], op[2], op[1],~op[0]);
    and (i_lw,   op[5],~op[4],~op[3],~op[2], op[1], op[0]);
    and (i_sw,   op[5],~op[4], op[3],~op[2], op[1], op[0]);
    and (i_beq, ~op[5],~op[4],~op[3], op[2],~op[1],~op[0]);
    and (i_bne, ~op[5],~op[4],~op[3], op[2],~op[1], op[0]);
    and (i_lui, ~op[5],~op[4], op[3], op[2], op[1], op[0]);
    and (i_j,   ~op[5],~op[4],~op[3],~op[2], op[1],~op[0]);      // i format
    and (i_jal, ~op[5],~op[4],~op[3],~op[2], op[1], op[0]);
    assign i_rs = i_add  | i_sub | i_and  | i_or  | i_xor | i_jr  | i_addi |
                  i_andi | i_ori | i_xori | i_lw  | i_sw  | i_beq | i_bne;
    assign i_rt = i_add  | i_sub | i_and  | i_or  | i_xor | i_sll | i_srl  |
                  i_sra  | i_sw  | i_beq  | i_bne | i_mtc0;    // mtc0 added
    assign wpcir = ~(ewreg & em2reg & (ern != 0) & (i_rs & (ern == rs) |
                                                    i_rt & (ern == rt)));
    always @ (ewreg or mwreg or ern or mrn or em2reg or mm2reg or rs or rt)
        begin
            fwda = 2'b00;                             // default: no hazards
            if (ewreg & (ern != 0) & (ern == rs) & ~em2reg) begin
                fwda = 2'b01;                         // select exe_alu
            end else begin
                if (mwreg & (mrn != 0) & (mrn == rs) & ~mm2reg) begin
                    fwda = 2'b10;                     // select mem_alu
                end else begin
                    if (mwreg & (mrn != 0) & (mrn == rs) & mm2reg) begin
                        fwda = 2'b11;                 // select mem_lw
                    end 
                end
            end
            fwdb = 2'b00;                             // default: no hazards
            if (ewreg & (ern != 0) & (ern == rt) & ~em2reg) begin
                fwdb = 2'b01;                         // select exe_alu
            end else begin
                if (mwreg & (mrn != 0) & (mrn == rt) & ~mm2reg) begin
                    fwdb = 2'b10;                     // select mem_alu
                end else begin
                    if (mwreg & (mrn != 0) & (mrn == rt) & mm2reg) begin
                        fwdb = 2'b11;                 // select mem_lw
                    end 
                end
            end
        end
    assign wmem     = i_sw & wpcir & ~ecancel & ~exc_ovr & ~mexc_ovr;
    assign regrt    = i_addi|i_andi|i_ori |i_xori|i_lw |i_lui |i_mfc0;
    assign jal      = i_jal;
    assign m2reg    = i_lw;
    assign shift    = i_sll |i_srl |i_sra;
    assign aluimm   = i_addi|i_andi|i_ori |i_xori|i_lw |i_lui |i_sw;
    assign sext     = i_addi|i_lw  |i_sw  |i_beq |i_bne;
    assign aluc[3]  = i_sra;
    assign aluc[2]  = i_sub |i_or  |i_srl |i_sra |i_ori |i_lui;
    assign aluc[1]  = i_xor |i_sll |i_srl |i_sra |i_xori|i_beq |i_bne|i_lui;
    assign aluc[0]  = i_and |i_or  |i_sll |i_srl |i_sra |i_andi|i_ori;
    assign pcsrc[1] = i_jr  |i_j   |i_jal;
    assign pcsrc[0] = i_beq & rsrtequ |i_bne & ~rsrtequ | i_j | i_jal;
    assign wreg     =(i_add |i_sub |i_and |i_or  |i_xor|i_sll  |
                      i_srl |i_sra |i_addi|i_andi|i_ori|i_xori |
                      i_lw  |i_lui |i_jal |i_mfc0) &  // mfc0 added
                      wpcir & ~ecancel & ~exc_ovr & ~mexc_ovr;
endmodule

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

// soy
module dff32 #(parameter WIDTH = 32)(input logic clk,clrn, // inputs d, clk, clrn (active low)
    input logic [WIDTH-1:0] d,
    output logic [WIDTH-1:0] q);            // dff with asynchronous reset
    always @ (posedge clk or negedge clrn) begin // always block, "or"
        if (!clrn) q <= 0;            // if clrn is asserted, reset dff
        else       q <= d;            // else store d to dff
    end
endmodule



module pipelined_cpu_exc_int2 (
    input logic clk, clrn, intr,
    output logic [31:0] pc, inst, ealu, malu, wdi,
    output logic inta
);
    assign base = 32'h00000008; // exception handler entry

    // fetch
    logic [31:0] pc4, npc, ins;

    assign pc4 = pc + 32'd4;
    pl_exc_i_mem _imem(.a(pc), .inst(ins));
    mux4x32 _npc(.a0(pc4), .a1(bpc), .a2(da), .a3({dpc4[31:28], jpc}), .s(pcsrc), .y(npc));
    mux4x32 _next_pc(.a0(npc), .a1(epc), .a2(base), .a3(32'b0), .s(selpc), .y(next_pc));
    dffe32 _pc(.clk(clk), .clrn(clrn), .e(wpcir), .d(next_pc), .q(pc));

    // fetch regs
    dffe32 _inst(.clk(clk), .clrn(clrn), .e(wpcir), .d(ins), .q(inst));
    dffe32 _dpc4(.clk(clk), .clrn(clrn), .e(wpcir), .d(pc4), .q(dpc4));
    dff _irq(.clk(clk), .clrn(clrn), .d(intr), .q(irq)); // this doesnt get stall tho

    dffe32 _pcd(.clk(clk), .clrn(clrn), .e(wpcir), .d(pc), .q(pcd)); // this also gets the stall write signal




    // decode 
    logic [5:0] op;
    logic [4:0] rs;
    logic [4:0] rt;
    logic [4:0] rd;
    logic [5:0] func;
    logic [15:0] imm;
    logic [25:0] addr;

    logic rsrtequ;
    logic [4:0] drn;
    logic [15:0] s16;
    logic [27:0] jpc;
    logic [31:0] dpc4, bpc, da, db, dimm, _qa, _qb;

    assign op = inst[31:26];
    assign rs = inst[25:21];
    assign rt = inst[20:16];
    assign rd = inst[15:11];
    assign func = inst[5:0];
    assign imm = inst[15:0];
    assign addr = inst[25:0];

    assign s16 = {16{sext & imm[15]}};
    assign rsrtequ = (da == db);
    assign jpc = {addr, 2'b00};
    assign dimm = {s16, imm};
    assign bpc = dpc4 + {dimm[29:0], 2'b00};

    mux4x32 __epcin(.a0(pc), .a1(pcd), .a2(pce), .a3(pcm), .s(sepc), .y(epcin));

    mux2x5 _drn(.a0(rd), .a1(rt), .s(regrt), .y(drn));
    mux4x32 _da(.a0(_qa), .a1(ealu), .a2(malu), .a3(mmo), .s(fwda), .y(da));
    mux4x32 _db(.a0(_qb), .a1(ealu), .a2(malu), .a3(mmo), .s(fwdb), .y(db));
    regfile _regfile(.clk(~clk), .clrn(clrn), .rna(rs), .rnb(rt), .d(wdi), .wn(wrn), .we(wwreg), .qa(_qa), .qb(_qb));

    assign sta_left = {sta[27:0], 4'b0};
    assign sta_right = {4'b0, sta[31:4]};
    mux2x32 _stalr(.a0(sta_right), .a1(sta_left), .s(exc), .y(stalr));
    mux2x32 _sta_in(.a0(stalr), .a1(db), .s(mtc0), .y(sta_in));
    mux2x32 _cau_in(.a0(cause), .a1(db), .s(mtc0), .y(cau_in));
    mux2x32 _epc_in(.a0(epcin), .a1(db), .s(mtc0), .y(epc_in));
    dffe32 _sta(.clk(clk), .clrn(clrn), .e(wsta), .d(sta_in), .q(sta));
    dffe32 _cau(.clk(clk), .clrn(clrn), .e(wcau), .d(cau_in), .q(cau));
    dffe32 _epc(.clk(clk), .clrn(clrn), .e(wepc), .d(epc_in), .q(epc));

    // decode regs
    dff32 _eimm(.clk(clk), .clrn(clrn), .d(dimm), .q(eimm));
    dff32 _eb(.clk(clk), .clrn(clrn), .d(db), .q(eb));
    dff32 _ea(.clk(clk), .clrn(clrn), .d(da), .q(ea));
    dff32 _epc4(.clk(clk), .clrn(clrn), .d(dpc4), .q(epc4));
    dff32 #(5) _ern0(.clk(clk), .clrn(clrn), .d(drn), .q(ern0));
    dff32 #(4) _ealuc(.clk(clk), .clrn(clrn), .d(aluc), .q(ealuc));
    dff _eshift(.clk(clk), .clrn(clrn), .d(shift), .q(eshift));
    dff _ealuimm(.clk(clk), .clrn(clrn), .d(aluimm), .q(ealuimm));
    dff _ejal(.clk(clk), .clrn(clrn), .d(jal), .q(ejal));
    dff _ewmem(.clk(clk), .clrn(clrn), .d(wmem), .q(ewmem));
    dff _em2reg(.clk(clk), .clrn(clrn), .d(m2reg), .q(em2reg));


    logic ewreg0, eove;
    assign exc_ovr = ov & eove;
    dff _ewreg(.clk(clk), .clrn(clrn), .d(wreg), .q(ewreg0));
    assign ewreg = ewreg0 & ~exc_ovr; 
    dff32 _pce(.clk(clk), .clrn(clrn), .d(pcd), .q(pce));

    dff _eove(.clk(clk), .clrn(clrn), .d(ove), .q(eove));
    dff _mexc_ove(.clk(clk), .clrn(clrn), .d(exc_ovr), .q(mexc_ovr));
    dff _eisbr(.clk(clk), .clrn(clrn), .d(is_branch), .q(eis_branch));
    dff _misbr(.clk(clk), .clrn(clrn), .d(eis_branch), .q(mis_branch));
    dff _ecancel(.clk(clk), .clrn(clrn), .d(cancel), .q(ecancel));



    // NEW SIGNALS 
    logic [31:0] epc, base, next_pc, pcd, pce, pcm, epcin, sta_right, sta_left, stalr, sta_in, cau_in, epc_in, pc8c0r;
    logic [31:0] cau;
    logic [1:0] emfc0;
    
    dff32 #(2) _emfc0(.clk(clk), .clrn(clrn), .d(mfc0), .q(emfc0));

    // exe 
    logic [31:0] eimm, eb, ea, epc4, epc8, _alua, _alub, ealu0, _esa;
    logic [4:0] ern0, ern;
    logic [3:0] ealuc;
    logic z, ov; // alu zero flag, not used since doing branch pred in decode stage
    logic eshift, ealuimm, ejal, ewmem, em2reg, ewreg;

    mux4x32 _pc8c0r(.a0(epc8), .a1(sta), .a2(cau), .a3(epc), .s(emfc0), .y(pc8c0r));
    mux2x32 _ealu(.a0(ealu0), .a1(pc8c0r), .s(ejal|emfc0[1]|emfc0[0]), .y(ealu));

    assign epc8 = epc4 + 32'd4;
    assign ern = ern0 | {5{ejal}};
    assign _esa = {eimm[5:0],eimm[31:6]}; // shift amount

    mux2x32 __alua(.a0(ea), .a1(_esa), .s(eshift), .y(_alua));
    mux2x32 __alub(.a0(eb), .a1(eimm), .s(ealuimm), .y(_alub));
    alu_ov _alu(.a(_alua), .b(_alub), .aluc(ealuc), .r(ealu0), .z(z), .v(ov)); 

    // exe registers
    dff32 __di(.clk(clk), .clrn(clrn), .d(eb), .q(_di));
    dff32 _malu(.clk(clk), .clrn(clrn), .d(ealu), .q(malu));
    dff32 #(5) _mrn(.clk(clk), .clrn(clrn), .d(ern), .q(mrn));
    dff _mwmem(.clk(clk), .clrn(clrn), .d(ewmem), .q(mwmem));
    dff _mm2reg(.clk(clk), .clrn(clrn), .d(em2reg), .q(mm2reg));
    dff _mwreg(.clk(clk), .clrn(clrn), .d(ewreg), .q(mwreg));

    dff32 _pcm(.clk(clk), .clrn(clrn), .d(pce), .q(pcm));






    // mem
    logic mwmem, mm2reg, mwreg;
    logic [31:0] mmo, _di;
    logic [4:0] mrn;

    pl_exc_d_mem _dmem(.clk(clk), .addr(malu), .datain(_di), .we(mwmem), .dataout(mmo));

    // mem registers
    dff32 __walu(.clk(clk), .clrn(clrn), .d(malu), .q(walu));
    dff32 __wmo(.clk(clk), .clrn(clrn), .d(mmo), .q(_wmo));
    dff32 #(5) _wrn(.clk(clk), .clrn(clrn), .d(mrn), .q(wrn));
    dff _wm2reg(.clk(clk), .clrn(clrn), .d(mm2reg), .q(wm2reg));
    dff _wwreg(.clk(clk), .clrn(clrn), .d(mwreg), .q(wwreg));




    // wb
    logic wwreg;
    logic [31:0] walu, _wmo;
    logic [4:0] wrn;
    
    mux2x32 _wdi(.a0(walu), .a1(_wmo), .s(wm2reg), .y(wdi)); // mmo  -> _wmo  1 ;; malu -> walu 0

    // control unit
    logic  [31:0] sta;                // status: IM[3:0]: ov,unimpl,sys,int 
    //logic   [5:0] op,func;
    //logic   [4:0] mrn,ern,rs,rt,rd;
    logic   [4:0] op1;                // for decode mfc0, mtc0, and eret
    assign op1 = rs; // idk why he renamed it but this is the argument to his cu
    //logic         mwreg,ewreg,em2reg,mm2reg,rsrtequ;
    logic         irq;                // interrupt request
    logic         ecancel;            // cancel in EXE stage
    logic         eis_branch;         // is_branch in EXE stage
    logic         mis_branch;         // is_branch in MEM stage
    logic         exc_ovr;            // overflow exception occurs
    logic         mexc_ovr;           // exc_ovr in MEM stage
    logic [31:0] cause;              // cause content
    logic  [3:0] aluc;
    logic  [1:0] pcsrc,fwda,fwdb;
    logic  [1:0] selpc;              // 00: npc;  01: epc; 10: exc_base
    logic  [1:0] mfc0;               // 00: epc8; 01: sta; 10: cau; 11: epc
    logic  [1:0] sepc;               // 00: pc;   01: pcd; 10: pce; 11: pcm
    logic        wpcir,wreg,m2reg,wmem,regrt,aluimm,sext,shift,jal;
    //logic        inta;               // interrupt acknowledgement
    logic        exc;                // any int or exc happened
    logic        mtc0;               // is mtc0 instruction
    logic        wsta;               // status register write enable
    logic        wcau;               // cause  register write enable
    logic        wepc;               // epc    register write enable
    logic        is_branch;          // is a branch or a jump
    logic        ove;                // ov enable = arith & sta[3]
    logic        cancel;             // exception cancels next instruction

    cu_exc_int _cu(.mwreg(mwreg),
        .mrn(mrn),
        .ern(ern),
        .ewreg(ewreg),
        .em2reg(em2reg),
        .mm2reg(mm2reg),
        .rsrtequ(rsrtequ),
        .func(func),
        .op(op),
        .rs(rs),
        .rt(rt),
        .rd(rd),
        .op1(op1),
        .wreg(wreg),
        .m2reg(m2reg),
        .wmem(wmem),
        .aluc(aluc),
        .regrt(regrt),
        .aluimm(aluimm),
        .fwda(fwda),
        .fwdb(fwdb),
        .wpcir(wpcir),
        .sext(sext),
        .pcsrc(pcsrc),
        .shift(shift),
        .jal(jal),
        .irq(irq),
        .sta(sta),
        .ecancel(ecancel),
        .eis_branch(eis_branch),
        .mis_branch(mis_branch),
        .inta(inta),
        .selpc(selpc),
        .exc(exc),
        .sepc(sepc),
        .cause(cause),
        .mtc0(mtc0),
        .wepc(wepc),
        .wcau(wcau),
        .wsta(wsta),
        .mfc0(mfc0),
        .is_branch(is_branch),
        .ove(ove),
        .cancel(cancel),
        .exc_ovr(exc_ovr),
        .mexc_ovr(mexc_ovr));

endmodule



// cache: direct-mapping, 64 block, 1 word/block, write-through
module d_cache(

    input logic clk, clrn,
    input logic [31:0] p_a,    // cpu addr
    input logic [31:0] p_dout, // cpu data out to mem
    input logic p_strobe, // cpu asking to read data
    input logic p_rw, // cpu is reading or writing
    input logic uncached, // I/O data should not be stored in cache
    output logic [31:0] p_din, // data sent to cpu
    output logic p_ready, // tell cpu data is ready 

    input logic m_ready, // mem ready
    input logic [31:0] m_dout, // data coming from memory
    output logic [31:0] m_a, // addr to memory
    output logic [31:0] m_din, // data sent to mem 
    output logic m_strobe, // asking to read mem
    output logic m_rw // whether we want to read/write mem
);
    
    // 64-item memory arrays: 1-bit valid bit, 24-bit tags, and 32-bit data. 
    logic d_valid [0:63]; 
    logic [23:0] d_tags [0:63]; 
    logic [31:0] d_data [0:63]; 
    // currently read out values of valid-bit, tag, data 
    logic valid = d_valid[index]; 
    logic [23:0] tagout = d_tags[index];
    logic [31:0] c_dout = d_data[index]; 

    // fields of address from cpu 
    logic [23:0] tag = p_a[31:8]; 
    logic [5:0] index = p_a[7:2]; 

    // update the memory. valid bits can be cleared with clrn
    always @(posedge clk or negedge clrn)
        if (!clrn) begin 
            for (integer i = 0; i < 64; i = i + 1)
                d_valid[i] <= 0; 
        end
        else if (c_write) // if writing to cache, valid bit at block index is 1
            d_valid[index] <= 1;
    always @ (posedge clk)
        if (c_write) begin // if writing to cache, write tag and data 
            d_tags[index] <= tag; 
            d_data[index] <= c_din; // data is either from cpu or mem
        end 
    
    // cache data in either from cpu or memory 
    logic [31:0] c_din; 
    assign c_din = p_rw ? p_dout : m_dout;

    // data sent to cpu is either from cache or mem
    assign p_din = cache_hit ? c_dout : m_dout; 

    // forward signals to the memory = write through policy
    assign m_din = p_dout; 
    assign m_a = p_a;    
    assign m_rw = p_rw;
    
    // straight forward
    logic c_write, cache_hit, cache_miss;
    assign cache_hit = p_strobe & valid & (tagout == tag); 
    assign cache_miss = p_strobe & (!valid | (tagout != tag)); 
    assign c_write = ~uncached & (p_rw | cache_miss & m_ready);

    // access memory when writing or cache miss
    assign m_strobe = p_rw | cache_miss; 

    // i guess cpu writing and mem ready = p_ready too
    // (cpu is reading and got cache hit) or (missed and mem is ready)
    assign p_ready = ~p_rw & cache_hit | 
                    (cache_miss | p_rw) & m_ready; 

endmodule

// READING
// basically we send a signal to memory asking for data at addr. 
// we send m_a, m_strobe, m_rw, and receive m_dout, m_ready (m_din sent for writes)
// on the way, we save this data in the cache. 
// if we request the same data again, we get it quicker from the cache. 

// WRITING 
// we write data to both memory and cache. 

// i was expecting some FSM that gets the next N words from memory when we request one word.


module slow_ram(
    input logic clk, clrn, 

    input logic [31:0] m_din, m_a, 
    input logic m_strobe, m_rw,

    output logic [31:0] m_dout, 
    output logic m_ready
);
    logic [31:0] data[0:127];

    // we strobe memory and write to it
    always @ (posedge clk) 
        if (m_strobe & m_rw) data[m_a] <= m_din;

    // the memory is ready after 6 clocks 
    assign m_dout = data[m_a];
    assign m_ready = (state_reg == ready);



    // counter fsm
    typedef enum {wait1, wait2, wait3, wait4, wait5, ready} state_type;
    state_type state_reg, state_nxt;

    always_ff @(posedge clk, negedge clrn)
        if (!clrn) state_reg <= wait1;
        else state_reg <= state_nxt;

    always_comb 
        case (state_reg)
            wait1 : if (m_strobe & ~m_rw) // if strobe and read, start state transition chain
                        state_nxt = wait2;
                    else 
                        state_nxt = wait1;
            wait2 : state_nxt = wait3;
            wait3 : state_nxt = wait4;
            wait4 : state_nxt = wait5;
            wait5 : state_nxt = ready;
            ready : if (!m_strobe | m_rw) // if not strobe or writing, go to start state
                        state_nxt = wait1;
                    else 
                        state_nxt = ready;
            default: state_nxt = wait1;
        endcase
endmodule
