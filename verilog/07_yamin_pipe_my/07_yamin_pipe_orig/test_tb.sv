`timescale 1ns/1ns

module testbench();

    reg         clk,clrn;
    wire [31:0] pc,inst,ealu,malu,wdi;
    pipelinedcpu cpu (clk,clrn,pc,inst,ealu,malu,wdi);
    initial begin
		// wave file
		$dumpfile("wave.vcd");
		$dumpvars(1, testbench);
		//$dumpvars(1, cpu);
        
             clrn = 0;
             clk  = 1;
        #1   clrn = 1;
        #335 $finish;
	end

    always #2 clk = !clk;
endmodule

