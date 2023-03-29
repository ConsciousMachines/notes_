

module box(
	input logic pl, pr, gl, gr,
	output logic p, g);

	assign p = pl & pr; // both left and right propagate the carry
	assign g = gl | (pl & gr); // either left generate or left propagates the right gen
endmodule

module test(
	input logic cin,
	input logic [7:0] a, b,
	output logic [7:0] s,
	output logic cout);

	logic [7:0] p, g;
	assign p = a | b; 
	assign g = a & b; 

	// level 1: boxes named box<level><num_from_right_to_left>
	logic p11,p12,p13,p14,g11,g12,g13,g14;
	box box11(p[0], 1'b0, g[0], cin, p11, g11); // prop from -1 is 0, gen from -1 is cin
	box box12(p[2], p[1], g[2], g[1], p12, g12);
	box box13(p[4], p[3], g[4], g[3], p13, g13);
	box box14(p[6], p[5], g[6], g[5], p14, g14);

	// level 2
	logic p21,p22,p23,p24,g21,g22,g23,g24;
	box box21(p[1], p11, g[1], g11, p21, g21);
	box box22(p12, p11, g12, g11, p22, g22);
	box box23(p[5], p13, g[5], g13, p23, g23);
	box box24(p14, p13, g14, g13, p24, g24);

	// level 3
	logic p31,p32,p33,p34,g31,g32,g33,g34;
	box box31(p[3], p22, g[3], g22, p31, g31);
	box box32(p13, p22, g13, g22, p32, g32);
	box box33(p23, p22, g23, g22, p33, g33);
	box box34(p24, p22, g24, g22, p34, g34);

	assign s = a ^ b ^ {g34, g33, g32, g31, g22, g21, g11, cin};
	logic meh; // throw out final propagate signal
	box box41(p[7], p34, g[7], g34, meh, cout);

endmodule
