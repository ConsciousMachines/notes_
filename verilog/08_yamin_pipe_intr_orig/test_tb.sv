`timescale 1ns/1ns

module testbench();

    reg         clk,clrn,intr;
    wire [31:0] pc,inst,ealu,malu,wdi;
    wire        inta;
    pipelined_cpu_exc_int cpu (clk,clrn,pc,inst,ealu,malu,wdi,intr,inta);

    initial begin
		// wave file
		$dumpfile("wave.vcd");
		//$dumpvars(1, testbench);
		$dumpvars(1, cpu);

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

