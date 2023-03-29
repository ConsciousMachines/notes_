`timescale 1ns/1ns

module testbench();


    logic clk, clrn;
    logic [31:0] m_din, m_a;
    logic m_strobe, m_rw;
    logic [31:0] m_dout;
    logic m_ready;

    slow_ram ram(.clk(clk), .clrn(clrn), .m_din(m_din), .m_a(m_a), 
        .m_strobe(m_strobe), .m_rw(m_rw), .m_dout(m_dout), .m_ready(m_ready));

    initial begin
		// wave file
		$dumpfile("wave.vcd");
		$dumpvars(1, ram);

        // initial setup
        m_din = 32'b0;
        m_a = 32'b0;
        m_strobe = 0;
        m_rw = 0;

             clrn = 0;
             clk  = 1;
        #1   clrn = 1;
        #1

        // write to memory (signals set up at time of clk posedge)
        m_din = 32'b10101;
        m_a = 32'b0;
        m_strobe = 1;
        m_rw = 1;
        #4

        m_strobe = 0;
        m_rw = 0;
        #1
        // now we read value
        m_strobe = 1;


        #40 $finish; 
    end

    always #2 clk = !clk;
    
endmodule

