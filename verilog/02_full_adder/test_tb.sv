`timescale 1ns / 1ps

module testbench();
	
	// bookkeeping
	logic [31:0] i, errs;
	logic [6:0] inputs[0:7];

	// signals
	logic cin, a, b, s, cout, p, g, exp_s, exp_cout, exp_p, exp_g;

	// instantiate UUT
	test uut(cin, a, b, s, cout, p, g);

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
		{cin, a, b, exp_s, exp_cout, exp_p, exp_g} = inputs[i];
		#1;
		if (s != exp_s || p != exp_p || g != exp_g || cout != exp_cout) begin // check result
			$display("Error: inputs = %b outputs = %b (%b expected)", {cin, a, b}, {s, cout, p, g}, {exp_s, exp_cout, exp_p, exp_g});
			errs = errs + 1;
		end
		i = i + 1;
		if (inputs[i] === 7'bx) begin 
			$display("%d tests completed with %d errors", i, errs);
			$finish;
		end
	end

endmodule

