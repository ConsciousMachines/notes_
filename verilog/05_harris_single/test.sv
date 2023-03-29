
module alu_dec(
	input logic [1:0] aluop,
	input logic [5:0] funct,
	output logic [2:0] alucontrol
);
	always_comb
		casex ({aluop, funct})
			8'b00_xxxxxx: alucontrol = 3'b010;
			8'bx1_xxxxxx: alucontrol = 3'b110;
			8'b1x_100000: alucontrol = 3'b010;
			8'b1x_100010: alucontrol = 3'b110;
			8'b1x_100100: alucontrol = 3'b000;
			8'b1x_100101: alucontrol = 3'b001;
			8'b1x_101010: alucontrol = 3'b111;
			default:      alucontrol = 3'bx; // shouldn't happen
		endcase 
endmodule


module main_dec(
	input logic [5:0] opcode,
	output logic [8:0] y
);
	always_comb
		casex (opcode)
			6'b000000: y = 9'b110000100;
			6'b100011: y = 9'b101001000;
			6'b101011: y = 9'b0x101x000;
			6'b000100: y = 9'b0x010x010;
			6'b001000: y = 9'b101000000;
			6'b000010: y = 9'b0xxx0xxx1;
			default:   y = 9'bx; // shouldn't happen
		endcase 
endmodule


module regfile(input  logic        clk, 
               input  logic        we3, 
               input  logic [4:0]  ra1, ra2, wa3, 
               input  logic [31:0] wd3, 
               output logic [31:0] rd1, rd2);

  logic [31:0] rf[31:0];

  always_ff @(posedge clk)
    if (we3) rf[wa3] <= wd3;	

  assign rd1 = (ra1 != 0) ? rf[ra1] : 0;
  assign rd2 = (ra2 != 0) ? rf[ra2] : 0;
endmodule


module adder(input  logic [31:0] a, b,
             output logic [31:0] y);

  assign y = a + b;
endmodule


module sl2 (input  logic [31:0] a,
           output logic [31:0] y);

  assign y = {a[29:0], 2'b00}; // shift left by 2
endmodule


module signext(input  logic [15:0] a,
               output logic [31:0] y);
              
  assign y = {{16{a[15]}}, a};
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


module alu(input  logic [31:0] a, b,
           input  logic [2:0]  alucontrol,
           output logic [31:0] result,
           output logic        zero);

  logic [31:0] condinvb, sum;

  assign condinvb = alucontrol[2] ? ~b : b;
  assign sum = a + condinvb + alucontrol[2];

  logic [31:0] soy = {32{sum[31]}};

  always_comb
    casex (alucontrol)
      3'bx00: result = a & b;
      3'bx01: result = a | b;
      3'bx10: result = sum;
      3'bx11: result = soy;
    endcase

  assign zero = (result == 32'b0);
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
	logic aluc0, aluc1, aluc2;
	logic Zero;
	logic MemtoReg, MemWrite, Branch, ALUSrc, RegDst, RegWrite, Jump;
	logic [31:0] PC, PCPlus4, PC_next, PC_nextt, PCBranch, PCJump, instr, WriteData;
	logic [31:0] SrcA, SrcB, ReadData, Result, ALUResult, SignImm, SignImm_sl2; 
	logic [1:0] _aluop;
	logic [2:0] ALUControl;
	logic [4:0] WriteReg;
	logic [8:0] _main_dec_y;

	// program counter
	flopr #(32) _PC(.clk(clk), .reset(reset), .d(PC_nextt), .q(PC));
	mux2 #(32) _PC_next(.d0(PCPlus4), .d1(PCBranch), .s(PCSrc), .y(PC_next));
	mux2 #(32) _PC_nextt(.d0(PC_next), .d1(PCJump), .s(Jump), .y(PC_nextt));
	adder _PCPlus4(.a(PC), .b(32'd4), .y(PCPlus4));
	assign PCSrc = Branch & Zero;
	assign PCJump = {PCPlus4[31:28], instr[25:0], 2'b0};

	// instruction mem
	imem _imem(.a(PC[7:2]), .rd(instr));

	// reg file
	mux2 #(5) _WriteReg(.d0(instr[20:16]), .d1(instr[15:11]), .s(RegDst), .y(WriteReg));
	regfile _regfile(.clk(clk), .ra1(instr[25:21]), .ra2(instr[20:16]), .wa3(WriteReg), 
		.wd3(Result), .we3(RegWrite), .rd1(SrcA), .rd2(WriteData));

	// sign extend
	signext _SignImm(.a(instr[15:0]), .y(SignImm));

	// PCBranch
	sl2 _SignImm_sl2(.a(SignImm), .y(SignImm_sl2));
	adder _PCBranch(.a(PCPlus4), .b(SignImm_sl2), .y(PCBranch));

	// ALU
	mux2 #(32) _SrcB(.d0(WriteData), .d1(SignImm), .s(ALUSrc), .y(SrcB));
	alu _alu(.a(SrcA), .b(SrcB), .result(ALUResult), .zero(Zero), .alucontrol(ALUControl));

	// data mem
	dmem _dmem(.clk(clk), .we(MemWrite), .a(ALUResult), .wd(WriteData), .rd(ReadData));
	mux2 #(32) _Result(.d0(ALUResult), .d1(ReadData), .s(MemtoReg), .y(Result));

	// Control Unit
	main_dec _main_dec(.opcode(instr[31:26]), .y(_main_dec_y));
	alu_dec _alu_dec(.aluop(_aluop), .funct(instr[5:0]), .alucontrol(ALUControl));
	assign {RegWrite, RegDst, ALUSrc, Branch, MemWrite, MemtoReg, _aluop, Jump} = _main_dec_y;

endmodule

