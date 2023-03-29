`timescale 1ns/1ns

module testbench();

    logic         clk,clrn;
    logic [31:0] pc,inst,ealu,malu,wdi;
    logic [31:0] pc_,inst_,ealu_,malu_,wdi_;
    pipelinedcpu cpu (clk,clrn,pc,inst,ealu,malu,wdi);
    pipelinedcpu2 cpu2 (clk,clrn,pc_,inst_,ealu_,malu_,wdi_);

    logic pc__,inst__,ealu__,malu__,wdi__;
    assign pc__ = (pc == pc_);
    assign inst__ = (inst == inst_);
    assign ealu__ = (ealu == ealu_);
    assign malu__ = (malu == malu_);
    assign wdi__ = (wdi == wdi_);
    
    initial begin
		// wave file
		$dumpfile("wave.vcd");
		
        //$dumpvars(2, cpu);
        //$dumpvars(1, cpu2);
		
        $dumpvars(1, testbench);
        
             clrn = 0;
             clk  = 1;
        #1   clrn = 1;
        #335 $finish;
	end

    always #2 clk = !clk;

endmodule

// bugs in my code:
// 1. regfile was not clocked with ~clk 
// 2. his shift amount input was {eimm[5:0],eimm[31:6]} while mine was just eimm. 
