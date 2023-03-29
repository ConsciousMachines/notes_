/*

cin a b  s cout p g
0 0 0 - 0 0 0 0
0 0 1 - 1 0 1 0
0 1 0 - 1 0 1 0
0 1 1 - 0 1 1 1
1 0 0 - 1 0 0 0
1 0 1 - 0 1 1 0
1 1 0 - 0 1 1 0
1 1 1 - 1 1 1 1


*/

module test(
	input logic cin, a, b,
	output logic s, cout, p, g);
	
	assign p = a | b;
	assign g = a & b;

	assign s = a ^ b ^ cin;
	assign cout = (a & b) | (a & cin) | (b & cin);
endmodule
