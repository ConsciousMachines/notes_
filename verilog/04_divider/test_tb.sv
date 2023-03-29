`timescale 1ns / 1ps

module testbench();
	
	// bookkeeping
	logic [31:0] i, errs;
	logic [31:0] inputs[0:65279];

	// signals
	logic clk, reset;
	logic [7:0] A, B;
	logic [7:0] mod, div, exp_mod, exp_div;
	logic done;

	// instantiate UUT
	test uut(clk, reset, A, B, mod, div, done);

	// initialize input and output files
	initial begin 
		// wave file
		$dumpfile("wave.vcd");
		$dumpvars(0, uut);

		// load test inputs file
		$readmemb("vectors.txt", inputs);
		i = 0; errs = 0;

		reset = 1;
		#1; // does pulse count as a START signal for the divide state machine?
		reset = 0;
	end

	// clock 
	always begin clk = 1; #1; clk = 0; #1; end

	// apply test vectors on rising edge of clk
	always begin 
		{A, B, exp_mod, exp_div} = inputs[i];
		#1;
		wait (done);
		#2;
		if (done) begin 
			if (mod != exp_mod || div != exp_div) begin // check result
				$display("Error: inputs = %b outputs = %b (%b expected)", {A,B}, {mod, div}, {exp_mod, exp_div});
				$display("%b_%b_%b_%b", A, B, exp_mod, exp_div);
				errs = errs + 1;
			end
			i = i + 1;
			if (inputs[i] === 32'bx) begin 
				$display("%d tests completed with %d errors", i, errs);
				$finish;
			end
		end
		// reset the divide state machine
		reset = 1;
		#1;
		reset = 0;
		#1;

	end

endmodule

