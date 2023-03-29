/*

unsigned addition overflows:
	when the carry out is set, so 1000 + 1000 = 1_0000 

signed addition overflow:
	adding two poz numbers and result > 2^(N-1) - 1
	adding two neg numbers and result < - 2^(N-1)
	overflow occurs if two numbers being added have same sign bit and result has diff sign bit

there must be a mistake in section 5.2.7 of Harris where he says D = R - B
	the difference D is negative = sign bit of D is 1
	but we are talking about unsigned numbers here 
	and 0001 - 1111 = 0001 + 1 + 0000 = 0010 does not have sign bit 1

*/

module test(
	input logic clk, reset,
	input logic [7:0] A, B,
	output logic [7:0] mod, div,
	output logic done
);
	logic [3:0] i; // state machine counter goes thru states 7,6,..0
	always_ff @ (posedge clk, posedge reset)
		if (reset) i <= 4'b0111; // state on first clock edge will be 0_111 = 7
		else if (~done) i <= i - 4'b0001;
	assign done = (i == 4'b1111); // signal that we passed state 0 and we are done


	logic [7:0] Rnext, R; // shift register that looks at leftmost 8-i digits of A at time i
	always_ff @ (posedge clk, posedge reset)
		if (reset) R <= 0;
		else if (~done) R <= does_fit ? Rnext - B : Rnext;
	assign Rnext = {R[6:0], A[i[2:0]]};


	logic does_fit; // TODO: use diff flags for signed/unsigned. im doing unsigned here
	assign does_fit = (Rnext >= B); // when B*factor fits into the slice and we write 1 in answer


	logic [7:0] Qnext, Q; // shift register for answer, we put in 1 or 0 each i depending on if D < 0
	always_ff @ (posedge clk, posedge reset)
		if (reset) Q <= 0;
		else if (~done) Q <= Qnext;
	assign Qnext = {Q[6:0], does_fit ? 1'b1 : 1'b0}; // next bit to insert in shift register Qnext


	assign mod = R;
	assign div = Q;
endmodule

