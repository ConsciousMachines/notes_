
module pipepc (npc,wpc,clk,clrn,pc);                      // program counter
    input         clk, clrn;                              // clock and reset
    input         wpc;                                    // pc write enable
    input  [31:0] npc;                                    // next pc
    output [31:0] pc;                                     // program counter
    // dffe32        (d,  clk,clrn,e,  q);
    dffe32 prog_cntr (npc,clk,clrn,wpc,pc);               // program counter
endmodule

module pipeif (pcsrc,pc,bpc,rpc,jpc,npc,pc4,ins);    // IF stage
    input  [31:0] pc;                                // program counter
    input  [31:0] bpc;                               // branch target
    input  [31:0] rpc;                               // jump target of jr
    input  [31:0] jpc;                               // jump target of j/jal
    input   [1:0] pcsrc;                             // next pc (npc) select
    output [31:0] npc;                               // next pc
    output [31:0] pc4;                               // pc + 4
    output [31:0] ins;                               // inst from inst mem
    mux4x32 next_pc (pc4,bpc,rpc,jpc,pcsrc,npc);     // npc select
    cla32  pc_plus4 (pc,32'h4,1'b0,pc4);             // pc + 4
    pl_inst_mem inst_mem (pc,ins);                   // inst mem
endmodule

module pipeir (pc4,ins,wir,clk,clrn,dpc4,inst);   // IF/ID pipeline register
    input         clk, clrn;                      // clock and reset
    input         wir;                            // write enable
    input  [31:0] pc4;                            // pc + 4 in IF stage
    input  [31:0] ins;                            // instruction in IF stage
    output [31:0] dpc4;                           // pc + 4 in ID stage
    output [31:0] inst;                           // instruction in ID stage
    // dffe32          (d,  clk,clrn,e,  q);
    dffe32 pc_plus4    (pc4,clk,clrn,wir,dpc4);   // pc+4 register
    dffe32 instruction (ins,clk,clrn,wir,inst);   // inst register
endmodule

module pipeid (mwreg,mrn,ern,ewreg,em2reg,mm2reg,dpc4,inst,wrn,wdi,ealu,
               malu,mmo,wwreg,clk,clrn,bpc,jpc,pcsrc,nostall,wreg,m2reg,
               wmem,aluc,aluimm,a,b,dimm,rn,shift,jal);// ID stage
    input         clk, clrn;                           // clock and reset
    input  [31:0] dpc4;                                // pc+4 in ID
    input  [31:0] inst;                                // inst in ID
    input  [31:0] wdi;                                 // data in WB
    input  [31:0] ealu;                                // alu res in EXE
    input  [31:0] malu;                                // alu res in MEM
    input  [31:0] mmo;                                 // mem out in MEM
    input   [4:0] ern;                                 // dest reg # in EXE
    input   [4:0] mrn;                                 // dest reg # in MEM
    input   [4:0] wrn;                                 // dest reg # in WB
    input         ewreg;                               // wreg in EXE
    input         em2reg;                              // m2reg in EXE
    input         mwreg;                               // wreg in MEM
    input         mm2reg;                              // m2reg in MEM
    input         wwreg;                               // wreg in MEM
    output [31:0] bpc;                                 // branch target
    output [31:0] jpc;                                 // jump target
    output [31:0] a, b;                                // operands a and b
    output [31:0] dimm;                                // 32-bit immediate
    output  [4:0] rn;                                  // dest reg #
    output  [3:0] aluc;                                // alu control
    output  [1:0] pcsrc;                               // next pc select
    output        nostall;                             // no pipeline stall
    output        wreg;                                // write regfile
    output        m2reg;                               // mem to reg
    output        wmem;                                // write memory
    output        aluimm;                              // alu input b is imm
    output        shift;                               // inst is a shift
    output        jal;                                 // inst is jal
    wire    [5:0] op   = inst[31:26];                  // op
    wire    [4:0] rs   = inst[25:21];                  // rs
    wire    [4:0] rt   = inst[20:16];                  // rt
    wire    [4:0] rd   = inst[15:11];                  // rd
    wire    [5:0] func = inst[05:00];                  // func
    wire   [15:0] imm  = inst[15:00];                  // immediate
    wire   [25:0] addr = inst[25:00];                  // address
    wire          regrt;                               // dest reg # is rt
    wire          sext;                                // sign extend
    wire   [31:0] qa, qb;                              // regfile outputs
    wire    [1:0] fwda, fwdb;                          // forward a and b
    wire   [15:0] s16  = {16{sext & inst[15]}};        // 16-bit signs
    wire   [31:0] dis  = {dimm[29:0],2'b00};           // branch offset
    wire          rsrtequ = ~|(a^b);                   // reg[rs] == reg[rt]
    pipeidcu cu (mwreg,mrn,ern,ewreg,em2reg,mm2reg,    // control unit
                 rsrtequ,func,op,rs,rt,wreg,m2reg,
                 wmem,aluc,regrt,aluimm,fwda,fwdb,
                 nostall,sext,pcsrc,shift,jal);
    regfile r_f (rs,rt,wdi,wrn,wwreg,~clk,clrn,qa,qb); // register file
    mux2x5  d_r (rd,rt,regrt,rn);                      // select dest reg #
    mux4x32 s_a (qa,ealu,malu,mmo,fwda,a);             // forward for a
    mux4x32 s_b (qb,ealu,malu,mmo,fwdb,b);             // forward for b
    cla32 b_adr (dpc4,dis,1'b0,bpc);                   // branch target
    assign dimm = {s16,imm};                           // 32-bit imm
    assign jpc  = {dpc4[31:28],addr,2'b00};            // jump target
endmodule

module pipedereg (dwreg,dm2reg,dwmem,daluc,daluimm,da,db,dimm,drn,dshift,
                  djal,dpc4,clk,clrn,ewreg,em2reg,ewmem,ealuc,ealuimm,ea,
                  eb,eimm,ern,eshift,ejal,epc4); // ID/EXE pipeline register
    input         clk, clrn;                 // clock and reset
    input  [31:0] da, db;                    // a and b in ID stage
    input  [31:0] dimm;                      // immediate in ID stage
    input  [31:0] dpc4;                      // pc+4 in ID stage
    input   [4:0] drn;                       // register number in ID stage
    input   [3:0] daluc;                     // alu control in ID stage
    input         dwreg,dm2reg,dwmem,daluimm,dshift,djal;    // in ID stage
    output [31:0] ea, eb;                    // a and b in EXE stage
    output [31:0] eimm;                      // immediate in EXE stage
    output [31:0] epc4;                      // pc+4 in EXE stage
    output  [4:0] ern;                       // register number in EXE stage
    output  [3:0] ealuc;                     // alu control in EXE stage
    output        ewreg,em2reg,ewmem,ealuimm,eshift,ejal;    // in EXE stage
    reg    [31:0] ea, eb, eimm, epc4;
    reg     [4:0] ern;
    reg     [3:0] ealuc;
    reg           ewreg,em2reg,ewmem,ealuimm,eshift,ejal;
    always @(negedge clrn or posedge clk)
        if (!clrn) begin                     // clear
            ewreg   <= 0;             em2reg  <= 0;
            ewmem   <= 0;             ealuc   <= 0;
            ealuimm <= 0;             ea      <= 0;
            eb      <= 0;             eimm    <= 0;
            ern     <= 0;             eshift  <= 0;
            ejal    <= 0;             epc4    <= 0;
        end else begin                       // register
            ewreg   <= dwreg;         em2reg  <= dm2reg;
            ewmem   <= dwmem;         ealuc   <= daluc;
            ealuimm <= daluimm;       ea      <= da;
            eb      <= db;            eimm    <= dimm;
            ern     <= drn;           eshift  <= dshift;
            ejal    <= djal;          epc4    <= dpc4;
        end
endmodule

module pipeexe (ealuc,ealuimm,ea,eb,eimm,eshift,ern0,epc4,ejal,ern,ealu);
    input  [31:0] ea, eb;                            // all in EXE stage
    input  [31:0] eimm;                              // imm
    input  [31:0] epc4;                              // pc+4
    input   [4:0] ern0;                              // temporary dest reg #
    input   [3:0] ealuc;                             // aluc
    input         ealuimm;                           // aluimm
    input         eshift;                            // shift
    input         ejal;                              // jal
    output [31:0] ealu;                              // EXE stage result
    output  [4:0] ern;                               // dest reg #
    wire   [31:0] alua;                              // alu input a
    wire   [31:0] alub;                              // alu input b
    wire   [31:0] ealu0;                             // alu result
    wire   [31:0] epc8;                              // pc+8
    wire          z;                                 // alu z flag, not used
    wire   [31:0] esa = {eimm[5:0],eimm[31:6]};      // shift amount
    cla32   ret_addr (epc4,32'h4,1'b0,epc8);         // pc+8
    mux2x32 alu_in_a (ea,esa,eshift,alua);           // alu input a
    mux2x32 alu_in_b (eb,eimm,ealuimm,alub);         // alu input b
    mux2x32 save_pc8 (ealu0,epc8,ejal,ealu);         // alu result or pc+8
    assign ern = ern0 | {5{ejal}};                   // dest reg #, jal: 31
    alu al_unit (alua,alub,ealuc,ealu0,z);           // alu result, z flag
endmodule

module pipeemreg (ewreg,em2reg,ewmem,ealu,eb,ern,clk,clrn,mwreg,mm2reg,
                  mwmem,malu,mb,mrn);        // EXE/MEM pipeline register
    input         clk, clrn;                 // clock and reset
    input  [31:0] ealu;                      // alu control in EXE stage
    input  [31:0] eb;                        // b in EXE stage
    input   [4:0] ern;                       // register number in EXE stage
    input         ewreg,em2reg,ewmem;        // in EXE stage
    output [31:0] malu;                      // alu control in MEM stage
    output [31:0] mb;                        // b in MEM stage
    output  [4:0] mrn;                       // register number in MEM stage
    output        mwreg,mm2reg,mwmem;        // in MEM stage
    reg    [31:0] malu,mb;
    reg     [4:0] mrn;
    reg           mwreg,mm2reg,mwmem;
    always @(negedge clrn or posedge clk)
        if (!clrn) begin                     // clear
            mwreg  <= 0;              mm2reg <= 0;
            mwmem  <= 0;              malu   <= 0;
            mb     <= 0;              mrn    <= 0;
        end else begin                       // register
            mwreg  <= ewreg;          mm2reg <= em2reg;
            mwmem  <= ewmem;          malu   <= ealu;
            mb     <= eb;             mrn    <= ern;
        end
endmodule

module pipemem (we,addr,datain,clk,dataout);          // MEM stage
    input         clk;                                // clock
    input  [31:0] addr;                               // address
    input  [31:0] datain;                             // data in (to mem)
    input         we;                                 // memory write
    output [31:0] dataout;                            // data out (from mem)
    pl_data_mem dmem (clk,dataout,datain,addr,we);    // data memory
endmodule

module pipemwreg (mwreg,mm2reg,mmo,malu,mrn,clk,clrn,wwreg,wm2reg,wmo,walu,
                  wrn);                      // MEM/WB pipeline register
    input         clk, clrn;                 // clock and reset
    input  [31:0] mmo;                       // memory data out in MEM stage
    input  [31:0] malu;                      // alu control in MEM stage
    input   [4:0] mrn;                       // register number in MEM stage
    input         mwreg, mm2reg;             // in MEM stage
    output [31:0] wmo;                       // memory data out in WB stage
    output [31:0] walu;                      // alu control in WB stage
    output  [4:0] wrn;                       // register number in WB stage
    output        wwreg, wm2reg;             // in WB stage
    reg    [31:0] wmo, walu;
    reg     [4:0] wrn;
    reg           wwreg,wm2reg;
    always @(negedge clrn or posedge clk)
      if (!clrn) begin                       // clear
          wwreg <= 0;                 wm2reg <= 0;
          wmo   <= 0;                 walu   <= 0;
          wrn   <= 0;
      end else begin                         // register
          wwreg <= mwreg;             wm2reg <= mm2reg;
          wmo   <= mmo;               walu   <= malu;
          wrn   <= mrn;
      end
endmodule

module pipewb (walu,wmo,wm2reg,wdi);      // WB stage
    input  [31:0] walu;                   // alu result or pc+8 in WB stage
    input  [31:0] wmo;                    // data out (from mem) in WB stage
    input         wm2reg;                 // memory to register in WB stage
    output [31:0] wdi;                    // data to be written into regfile
    mux2x32 wb (walu,wmo,wm2reg,wdi);     // select for wdi
endmodule




module pipelinedcpu (clk,clrn,pc,inst,ealu,malu,wdi);       // pipelined cpu
    input         clk, clrn;    // clock and reset          // plus inst mem
    output [31:0] pc;           // program counter          // and  data mem
    output [31:0] inst;         // instruction in ID stage
    output [31:0] ealu;         // alu result in EXE stage
    output [31:0] malu;         // alu result in MEM stage
    output [31:0] wdi;          // data to be written into register file
    // signals in IF stage
    wire   [31:0] pc4;          // pc+4 in IF stage
    wire   [31:0] ins;          // instruction in IF stage
    wire   [31:0] npc;          // next pc in IF stage
    // signals in ID stage
    wire   [31:0] dpc4;         // pc+4 in ID stage
    wire   [31:0] bpc;          // branch target of beq and bne instructions
    wire   [31:0] jpc;          // jump target of jr instruction
    wire   [31:0] da,db;        // two operands a and b in ID stage
    wire   [31:0] dimm;         // 32-bit extended immediate in ID stage
    wire    [4:0] drn;          // destination register number in ID stage
    wire    [3:0] daluc;        // alu control in ID stage
    wire    [1:0] pcsrc;        // next pc (npc) select in ID stage
    wire          wpcir;        // pipepc and pipeir write enable
    wire          dwreg;        // register file write enable in ID stage
    wire          dm2reg;       // memory to register in ID stage
    wire          dwmem;        // memory write in ID stage
    wire          daluimm;      // alu input b is an immediate in ID stage
    wire          dshift;       // shift in ID stage
    wire          djal;         // jal in ID stage
    // signals in EXE stage
    wire   [31:0] epc4;         // pc+4 in EXE stage
    wire   [31:0] ea,eb;        // two operands a and b in EXE stage
    wire   [31:0] eimm;         // 32-bit extended immediate in EXE stage
    wire    [4:0] ern0;         // temporary register number in WB stage
    wire    [4:0] ern;          // destination register number in EXE stage
    wire    [3:0] ealuc;        // alu control in EXE stage
    wire          ewreg;        // register file write enable in EXE stage
    wire          em2reg;       // memory to register in EXE stage
    wire          ewmem;        // memory write in EXE stage
    wire          ealuimm;      // alu input b is an immediate in EXE stage
    wire          eshift;       // shift in EXE stage
    wire          ejal;         // jal in EXE stage
    // signals in MEM stage
    wire   [31:0] mb;           // operand b in MEM stage
    wire   [31:0] mmo;          // memory data out in MEM stage
    wire    [4:0] mrn;          // destination register number in MEM stage
    wire          mwreg;        // register file write enable in MEM stage
    wire          mm2reg;       // memory to register in MEM stage
    wire          mwmem;        // memory write in MEM stage
    // signals in WB stage
    wire   [31:0] wmo;          // memory data out in WB stage
    wire   [31:0] walu;         // alu result in WB stage
    wire    [4:0] wrn;          // destination register number in WB stage
    wire          wwreg;        // register file write enable in WB stage
    wire          wm2reg;       // memory to register in WB stage
    // program counter
    pipepc   prog_cnt (npc,wpcir,clk,clrn,pc);
    pipeif   if_stage (pcsrc,pc,bpc,da,jpc,npc,pc4,ins);        // IF stage
    // IF/ID pipeline register
    pipeir     fd_reg (pc4,ins,wpcir,clk,clrn,dpc4,inst);
    pipeid   id_stage (mwreg,mrn,ern,ewreg,em2reg,mm2reg,dpc4,inst,wrn,wdi,
                       ealu,malu,mmo,wwreg,clk,clrn,bpc,jpc,pcsrc,wpcir,
                       dwreg,dm2reg,dwmem,daluc,daluimm,da,db,dimm,drn,
                       dshift,djal);                            // ID stage
    // ID/EXE pipeline register
    pipedereg  de_reg (dwreg,dm2reg,dwmem,daluc,daluimm,da,db,dimm,drn, 
                       dshift,djal,dpc4,clk,clrn,ewreg,em2reg,ewmem,
                       ealuc,ealuimm,ea,eb,eimm,ern0,eshift,ejal,epc4);
    pipeexe exe_stage (ealuc,ealuimm,ea,eb,eimm,eshift,ern0,epc4,ejal,
                       ern,ealu);                               // EXE stage
    // EXE/MEM pipeline register
    pipeemreg  em_reg (ewreg,em2reg,ewmem,ealu,eb,ern,clk,clrn,mwreg,
                       mm2reg,mwmem,malu,mb,mrn);
    pipemem mem_stage (mwmem,malu,mb,clk,mmo);                  // MEM stage
    // MEM/WB pipeline register
    pipemwreg  mw_reg (mwreg,mm2reg,mmo,malu,mrn,clk,clrn,wwreg,wm2reg,
                       wmo,walu,wrn);
    pipewb   wb_stage (walu,wmo,wm2reg,wdi);                    // WB stage
endmodule

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

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

module pl_data_mem (clk,dataout,datain,addr,we); // data memory, ram
    input         clk;                     // clock
    input  [31:0] addr;                    // ram address
    input  [31:0] datain;                  // data in (to memory)
    input         we;                      // write enable
    output [31:0] dataout;                 // data out (from memory)
    reg    [31:0] ram [0:31];              // ram cells: 32 words * 32 bits
    assign dataout = ram[addr[6:2]];       // use 5-bit word address
    always @ (posedge clk) begin
        if (we) ram[addr[6:2]] = datain;   // write ram
    end
    integer i;
    initial begin                          // ram initialization
        for (i = 0; i < 32; i = i + 1)
            ram[i] = 0;
        // ram[word_addr] = data           // (byte_addr) item in data array
        ram[5'h14] = 32'h000000a3;         // (50) data[0]   0 +  a3 =  a3
        ram[5'h15] = 32'h00000027;         // (54) data[1]  a3 +  27 =  ca
        ram[5'h16] = 32'h00000079;         // (58) data[2]  ca +  79 = 143
        ram[5'h17] = 32'h00000115;         // (5c) data[3] 143 + 115 = 258
        // ram[5'h18] should be 0x00000258, the sum stored by sw instruction
    end
endmodule

module pl_inst_mem (a,inst);           // instruction memory, rom
    input  [31:0] a;                   // rom address
    output [31:0] inst;                // rom content = rom[a]
    wire   [31:0] rom [0:63];          // rom cells: 64 words * 32 bits
    // rom[word_addr] = instruction    // (pc) label   instruction
    assign rom[6'h00] = 32'h3c010000;  // (00) main:   lui  $1, 0         
    assign rom[6'h01] = 32'h34240050;  // (04)         ori  $4, $1, 80    
    assign rom[6'h02] = 32'h0c00001b;  // (08) call:   jal  sum           
    assign rom[6'h03] = 32'h20050004;  // (0c) dslot1: addi $5, $0,  4    
    assign rom[6'h04] = 32'hac820000;  // (10) return: sw   $2, 0($4)     
    assign rom[6'h05] = 32'h8c890000;  // (14)         lw   $9, 0($4)     
    assign rom[6'h06] = 32'h01244022;  // (18)         sub  $8, $9, $4    
    assign rom[6'h07] = 32'h20050003;  // (1c)         addi $5, $0,  3    
    assign rom[6'h08] = 32'h20a5ffff;  // (20) loop2:  addi $5, $5, -1    
    assign rom[6'h09] = 32'h34a8ffff;  // (24)         ori  $8, $5, 0xffff
    assign rom[6'h0a] = 32'h39085555;  // (28)         xori $8, $8, 0x5555
    assign rom[6'h0b] = 32'h2009ffff;  // (2c)         addi $9, $0, -1    
    assign rom[6'h0c] = 32'h312affff;  // (30)         andi $10,$9,0xffff
    assign rom[6'h0d] = 32'h01493025;  // (34)         or   $6, $10, $9   
    assign rom[6'h0e] = 32'h01494026;  // (38)         xor  $8, $10, $9   
    assign rom[6'h0f] = 32'h01463824;  // (3c)         and  $7, $10, $6   
    assign rom[6'h10] = 32'h10a00003;  // (40)         beq  $5, $0, shift 
    assign rom[6'h11] = 32'h00000000;  // (44) dslot2: nop                
    assign rom[6'h12] = 32'h08000008;  // (48)         j    loop2         
    assign rom[6'h13] = 32'h00000000;  // (4c) dslot3: nop                
    assign rom[6'h14] = 32'h2005ffff;  // (50) shift:  addi $5, $0, -1    
    assign rom[6'h15] = 32'h000543c0;  // (54)         sll  $8, $5, 15    
    assign rom[6'h16] = 32'h00084400;  // (58)         sll  $8, $8, 16    
    assign rom[6'h17] = 32'h00084403;  // (5c)         sra  $8, $8, 16    
    assign rom[6'h18] = 32'h000843c2;  // (60)         srl  $8, $8, 15    
    assign rom[6'h19] = 32'h08000019;  // (64) finish: j    finish        
    assign rom[6'h1a] = 32'h00000000;  // (68) dslot4: nop                
    assign rom[6'h1b] = 32'h00004020;  // (6c) sum:    add  $8, $0, $0    
    assign rom[6'h1c] = 32'h8c890000;  // (70) loop:   lw   $9, 0($4)     
    assign rom[6'h1d] = 32'h01094020;  // (74) stall:  add  $8, $8, $9    
    assign rom[6'h1e] = 32'h20a5ffff;  // (78)         addi $5, $5, -1    
    assign rom[6'h1f] = 32'h14a0fffc;  // (7c)         bne  $5, $0, loop  
    assign rom[6'h20] = 32'h20840004;  // (80) dslot5: addi $4, $4,  4    
    assign rom[6'h21] = 32'h03e00008;  // (84)         jr   $31           
    assign rom[6'h22] = 32'h00081000;  // (88) dslot6: sll  $2, $8, 0     
    assign inst = rom[a[7:2]];         // use 6-bit word address to read rom
endmodule

module pipeidcu (mwreg,mrn,ern,ewreg,em2reg,mm2reg,rsrtequ,func,op,rs,rt,
                 wreg,m2reg,wmem,aluc,regrt,aluimm,fwda,fwdb,nostall,sext,
                 pcsrc,shift,jal); // control unit in ID stage
    input   [5:0] op,func;  // op and func fields in instruction
    input   [4:0] rs,rt;    // rs and rt fields in instruction
    input   [4:0] ern;      // destination register number in EXE stage
    input   [4:0] mrn;      // destination register number in MEM stage
    input         ewreg;    // register file write enable in EXE stage
    input         em2reg;   // memory to register in EXE stage
    input         mwreg;    // register file write enable in MEM stage
    input         mm2reg;   // memory to register in MEM stage
    input         rsrtequ;  // reg[rs] == reg[rt]
    output  [3:0] aluc;     // alu control
    output  [1:0] pcsrc;    // next pc (npc) select
    output  [1:0] fwda;     // forward a: 00:qa; 01:exe; 10:mem; 11:mem_mem
    output  [1:0] fwdb;     // forward b: 00:qb; 01:exe; 10:mem; 11:mem_mem
    output        wreg;     // register file write enable
    output        m2reg;    // memory to register
    output        wmem;     // memory write
    output        aluimm;   // alu input b is an immediate
    output        shift;    // instruction in ID stage is a shift
    output        jal;      // instruction in ID stage is jal
    output        regrt;    // destination register number is rt
    output        sext;     // sign extend
    output        nostall;  // no stall (pipepc and pipeir write enable)
    // instruction decode
    wire rtype,i_add,i_sub,i_and,i_or,i_xor,i_sll,i_srl,i_sra,i_jr;
    wire i_addi,i_andi,i_ori,i_xori,i_lw,i_sw,i_beq,i_bne,i_lui,i_j,i_jal;
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
    and (i_j,   ~op[5],~op[4],~op[3],~op[2], op[1],~op[0]);      // j format
    and (i_jal, ~op[5],~op[4],~op[3],~op[2], op[1], op[0]);
    // instructions that use rs
    wire i_rs = i_add  | i_sub | i_and  | i_or  | i_xor | i_jr  | i_addi |
                i_andi | i_ori | i_xori | i_lw  | i_sw  | i_beq | i_bne;
    // instructions that use rt
    wire i_rt = i_add  | i_sub | i_and  | i_or  | i_xor | i_sll | i_srl  |
                i_sra  | i_sw  | i_beq  | i_bne;
    // pipeline stall caused by data dependency with lw instruction
    assign nostall = ~(ewreg & em2reg & (ern != 0) & (i_rs & (ern == rs) |
                                                      i_rt & (ern == rt)));
    reg [1:0] fwda, fwdb;  // forwarding, multiplexer's select signals
    always @ (ewreg, mwreg, ern, mrn, em2reg, mm2reg, rs, rt) begin
        // forward control signal for alu input a
        fwda = 2'b00;                                 // default: no hazards
        if (ewreg & (ern != 0) & (ern == rs) & ~em2reg) begin
            fwda = 2'b01;                             // select exe_alu
        end else begin
            if (mwreg & (mrn != 0) & (mrn == rs) & ~mm2reg) begin
                fwda = 2'b10;                         // select mem_alu
            end else begin
                if (mwreg & (mrn != 0) & (mrn == rs) & mm2reg) begin
                    fwda = 2'b11;                     // select mem_lw
                end 
            end
        end
        // forward control signal for alu input b
        fwdb = 2'b00;                                 // default: no hazards
        if (ewreg & (ern != 0) & (ern == rt) & ~em2reg) begin
            fwdb = 2'b01;                             // select exe_alu
        end else begin
            if (mwreg & (mrn != 0) & (mrn == rt) & ~mm2reg) begin
                fwdb = 2'b10;                         // select mem_alu
            end else begin
                if (mwreg & (mrn != 0) & (mrn == rt) & mm2reg) begin
                    fwdb = 2'b11;                     // select mem_lw
                end 
            end
        end
    end
    // control signals
    assign wreg     =(i_add |i_sub |i_and |i_or  |i_xor |i_sll |i_srl |
                      i_sra |i_addi|i_andi|i_ori |i_xori|i_lw  |i_lui |
                      i_jal)& nostall;       // prevent from executing twice
    assign regrt    = i_addi|i_andi|i_ori |i_xori|i_lw  |i_lui;
    assign jal      = i_jal;
    assign m2reg    = i_lw;
    assign shift    = i_sll |i_srl |i_sra;
    assign aluimm   = i_addi|i_andi|i_ori |i_xori|i_lw  |i_lui |i_sw;
    assign sext     = i_addi|i_lw  |i_sw  |i_beq |i_bne;
    assign aluc[3]  = i_sra;
    assign aluc[2]  = i_sub |i_or  |i_srl |i_sra |i_ori |i_lui;
    assign aluc[1]  = i_xor |i_sll |i_srl |i_sra |i_xori|i_beq |i_bne|i_lui;
    assign aluc[0]  = i_and |i_or  |i_sll |i_srl |i_sra |i_andi|i_ori;
    assign wmem     = i_sw  & nostall;       // prevent from executing twice
    assign pcsrc[1] = i_jr  |i_j   |i_jal;
    assign pcsrc[0] = i_beq & rsrtequ | i_bne & ~rsrtequ | i_j | i_jal;
endmodule

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------


module dff (d,clk,clrn,q);            // dff with asynchronous reset
    input      d, clk, clrn;          // inputs d, clk, clrn (active low)
    output reg q;                     // output q, register type
    always @ (posedge clk or negedge clrn) begin // always block, "or"
        if (!clrn) q <= 0;            // if clrn is asserted, reset dff
        else       q <= d;            // else store d to dff
    end
endmodule
// soy
module dff32 (input logic clk,clrn, // inputs d, clk, clrn (active low)
    input logic [31:0] d,
    output logic [31:0] q);            // dff with asynchronous reset
    always @ (posedge clk or negedge clrn) begin // always block, "or"
        if (!clrn) q <= 0;            // if clrn is asserted, reset dff
        else       q <= d;            // else store d to dff
    end
endmodule
// soy
module dff5 (input logic clk,clrn, // inputs d, clk, clrn (active low)
    input logic [4:0] d,
    output logic [4:0] q);            // dff with asynchronous reset
    always @ (posedge clk or negedge clrn) begin // always block, "or"
        if (!clrn) q <= 0;            // if clrn is asserted, reset dff
        else       q <= d;            // else store d to dff
    end
endmodule
// soy
module dff4 (input logic clk,clrn, // inputs d, clk, clrn (active low)
    input logic [3:0] d,
    output logic [3:0] q);            // dff with asynchronous reset
    always @ (posedge clk or negedge clrn) begin // always block, "or"
        if (!clrn) q <= 0;            // if clrn is asserted, reset dff
        else       q <= d;            // else store d to dff
    end
endmodule



module pipelinedcpu2 (
    input logic clk, clrn,
    output logic [31:0] pc, inst, ealu, malu, wdi
);

    // fetch
    logic [31:0] pc4, npc, ins;

    assign pc4 = pc + 32'd4;
    dffe32 _pc(.clk(clk), .clrn(clrn), .e(wpcir), .d(npc), .q(pc));
    mux4x32 _npc(.a0(pc4), .a1(bpc), .a2(da), .a3({dpc4[31:28], jpc}), .s(pcsrc), .y(npc));
    pl_inst_mem _imem(.a(pc), .inst(ins));

    // fetch regs
    dffe32 _inst(.clk(clk), .clrn(clrn), .e(wpcir), .d(ins), .q(inst));
    dffe32 _dpc4(.clk(clk), .clrn(clrn), .e(wpcir), .d(pc4), .q(dpc4));




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

    mux2x5 _drn(.a0(rd), .a1(rt), .s(regrt), .y(drn));
    mux4x32 _da(.a0(_qa), .a1(ealu), .a2(malu), .a3(mmo), .s(fwda), .y(da));
    mux4x32 _db(.a0(_qb), .a1(ealu), .a2(malu), .a3(mmo), .s(fwdb), .y(db));
    regfile _regfile(.clk(~clk), .clrn(clrn), .rna(rs), .rnb(rt), .d(wdi), .wn(wrn), .we(wwreg), .qa(_qa), .qb(_qb));

    // decode regs
    dff32 _eimm(.clk(clk), .clrn(clrn), .d(dimm), .q(eimm));
    dff32 _eb(.clk(clk), .clrn(clrn), .d(db), .q(eb));
    dff32 _ea(.clk(clk), .clrn(clrn), .d(da), .q(ea));
    dff32 _epc4(.clk(clk), .clrn(clrn), .d(dpc4), .q(epc4));
    dff5 _ern0(.clk(clk), .clrn(clrn), .d(drn), .q(ern0));
    dff4 _ealuc(.clk(clk), .clrn(clrn), .d(aluc), .q(ealuc));
    dff _eshift(.clk(clk), .clrn(clrn), .d(shift), .q(eshift));
    dff _ealuimm(.clk(clk), .clrn(clrn), .d(aluimm), .q(ealuimm));
    dff _ejal(.clk(clk), .clrn(clrn), .d(jal), .q(ejal));
    dff _ewmem(.clk(clk), .clrn(clrn), .d(wmem), .q(ewmem));
    dff _em2reg(.clk(clk), .clrn(clrn), .d(m2reg), .q(em2reg));
    dff _ewreg(.clk(clk), .clrn(clrn), .d(wreg), .q(ewreg));




    // exe 
    logic [31:0] eimm, eb, ea, epc4, epc8, _alua, _alub, _aluresult, _esa;
    logic [4:0] ern0, ern;
    logic [3:0] ealuc;
    logic _z; // alu zero flag, not used since doing branch pred in decode stage
    logic eshift, ealuimm, ejal, ewmem, em2reg, ewreg;

    assign epc8 = epc4 + 32'd4;
    assign ern = ern0 | {5{ejal}};
    assign _esa = {eimm[5:0],eimm[31:6]}; // shift amount

    mux2x32 __alua(.a0(ea), .a1(_esa), .s(eshift), .y(_alua));
    mux2x32 __alub(.a0(eb), .a1(eimm), .s(ealuimm), .y(_alub));
    mux2x32 _ealu(.a0(_aluresult), .a1(epc8), .s(ejal), .y(ealu));
    alu _alu(.a(_alua), .b(_alub), .aluc(ealuc), .r(_aluresult));

    // exe registers
    dff32 __di(.clk(clk), .clrn(clrn), .d(eb), .q(_di));
    dff32 _malu(.clk(clk), .clrn(clrn), .d(ealu), .q(malu));
    dff5 _mrn(.clk(clk), .clrn(clrn), .d(ern), .q(mrn));
    dff _mwmem(.clk(clk), .clrn(clrn), .d(ewmem), .q(mwmem));
    dff _mm2reg(.clk(clk), .clrn(clrn), .d(em2reg), .q(mm2reg));
    dff _mwreg(.clk(clk), .clrn(clrn), .d(ewreg), .q(mwreg));




    // mem
    logic mwmem, mm2reg, mwreg;
    logic [31:0] mmo, _di;
    logic [4:0] mrn;

    pl_data_mem _dmem(.clk(clk), .addr(malu), .datain(_di), .we(mwmem), .dataout(mmo));

    // mem registers
    dff32 __walu(.clk(clk), .clrn(clrn), .d(malu), .q(_walu));
    dff32 __wmo(.clk(clk), .clrn(clrn), .d(mmo), .q(_wmo));
    dff5 _wrn(.clk(clk), .clrn(clrn), .d(mrn), .q(wrn));
    dff _wm2reg(.clk(clk), .clrn(clrn), .d(mm2reg), .q(wm2reg));
    dff _wwreg(.clk(clk), .clrn(clrn), .d(mwreg), .q(wwreg));




    // wb
    logic wwreg;
    logic [31:0] _walu, _wmo;
    logic [4:0] wrn;
    
    mux2x32 _wdi(.a0(_walu), .a1(_wmo), .s(wm2reg), .y(wdi)); // mmo  -> _wmo  1 ;; malu -> _walu 0

    // control unit
    logic wpcir, wreg, m2reg, wmem, jal, aluimm, shift, regrt, sext;
    logic [1:0] fwda, fwdb, pcsrc;
    logic [3:0] aluc;
    pipeidcu _cu(.mwreg(mwreg),
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
                    .wreg(wreg),
                    .m2reg(m2reg),
                    .wmem(wmem),
                    .aluc(aluc),
                    .regrt(regrt),
                    .aluimm(aluimm),
                    .fwda(fwda),
                    .fwdb(fwdb),
                    .nostall(wpcir),
                    .sext(sext),
                    .pcsrc(pcsrc),
                    .shift(shift),
                    .jal(jal));

endmodule
















