`timescale 1ns / 1ps

module testbench();
	
	// bookkeeping
	logic [31:0] i, errs;
	logic [2:0] inputs[0:3];

	// signals
	logic a, b, y, exp;

	// instantiate UUT
	test uut(a, b, y);

	// initialize input and output files
	initial begin 
		// load test inputs file
		$readmemb("vectors.txt", inputs);
		i = 0; errs = 0;

		// wave file
		$dumpfile("wave.vcd");
		$dumpvars(0, uut);
	end

	// apply test vectors on rising edge of clk
	always begin 
		{a, b, exp} = inputs[i];
		#1;
		if (y != exp) begin // check result
			$display("Error: inputs = %b outputs = %b (%b expected)", {a, b}, y, exp);
			errs = errs + 1;
		end
		i = i + 1;
		if (inputs[i] === 3'bx) begin 
			$display("%d tests completed with %d errors", i, errs);
			$finish;
		end
	end

endmodule

