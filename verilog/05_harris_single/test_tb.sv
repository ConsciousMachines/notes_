`timescale 1ns / 1ps

module testbench();
	
	logic [31:0] i;

	// signals
	logic clk, reset;

	// instantiate UUT
	test uut(clk, reset);

	// initialize input and output files
	initial begin 
		// wave file
		$dumpfile("wave.vcd");
		$dumpvars(0, uut);


		reset = 1;
		#1; // does pulse count as a START signal for the divide state machine?
		reset = 0;

		#100 $finish;
	end

	// clock 
	always begin clk = 1; #1; clk = 0; #1; end

endmodule

