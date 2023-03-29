`timescale 1ns / 1ps

module testbench();
	
	// bookkeeping
	logic [31:0] i, errs;
	logic [25:0] inputs[0:131071];

	// signals
	logic cin;
	logic [7:0] a, b;
	logic [7:0] s, exp_s;
	logic cout, exp_cout;

	// instantiate UUT
	test uut(cin, a, b, s, cout);

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
		{cin, a, b, exp_s, exp_cout} = inputs[i];
		#1;
		if (s != exp_s || cout != exp_cout) begin // check result
			$display("Error: inputs = %b outputs = %b (%b expected)", {cin, a, b}, {s, cout}, {exp_s, exp_cout});
			errs = errs + 1;
		end
		i = i + 1;
		if (inputs[i] === 26'bx) begin 
			$display("%d tests completed with %d errors", i, errs);
			$finish;
		end
	end

endmodule

