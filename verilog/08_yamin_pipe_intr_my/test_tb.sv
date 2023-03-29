`timescale 1ns/1ns

module testbench();

    reg         clk,clrn,intr;
    wire [31:0] pc,inst,ealu,malu,wdi, pc_,inst_,ealu_,malu_,wdi_;
    wire        inta, inta_;

    pipelined_cpu_exc_int cpu (.clk(clk), .clrn(clrn), .pc(pc), .inst(inst), 
        .ealu(ealu), .malu(malu), .wdi(wdi), .intr(intr), .inta(inta));

    pipelined_cpu_exc_int2 cpu2 (.clk(clk), .clrn(clrn), .pc(pc_), .inst(inst_), 
        .ealu(ealu_), .malu(malu_), .wdi(wdi_), .intr(intr), .inta(inta_));
    logic pc__,inst__,ealu__,malu__,wdi__,inta__;
    assign pc__ = (pc == pc_);
    assign inst__ = (inst == inst_);
    assign ealu__ = (ealu == ealu_);
    assign malu__ = (malu == malu_);
    assign wdi__ = (wdi == wdi_);
    assign inta__ = (inta == inta_);
    


    initial begin
		// wave file
		$dumpfile("wave.vcd");
		$dumpvars(1, testbench);
		//$dumpvars(1, cpu2);

             clrn = 0;
             clk  = 1;
             intr = 0;
        #1   clrn = 1;
        #149 intr = 1;
        #8   intr = 0;
        #56  intr = 1;
        #8   intr = 0;
        #88  intr = 1;
        #8   intr = 0;
        #142 $finish; 
    end

    always #2 clk = !clk;
    
endmodule

