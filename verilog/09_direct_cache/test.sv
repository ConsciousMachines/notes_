

// cache: direct-mapping, 64 block, 1 word/block, write-through
module d_cache(

    input logic clk, clrn,
    input logic [31:0] p_a,    // cpu addr
    input logic [31:0] p_dout, // cpu data out to mem
    input logic p_strobe, // cpu asking to read data
    input logic p_rw, // cpu is reading or writing
    input logic uncached, // I/O data should not be stored in cache
    output logic [31:0] p_din, // data sent to cpu
    output logic p_ready, // tell cpu data is ready 

    input logic m_ready, // mem ready
    input logic [31:0] m_dout, // data coming from memory
    output logic [31:0] m_a, // addr to memory
    output logic [31:0] m_din, // data sent to mem 
    output logic m_strobe, // asking to read mem
    output logic m_rw // whether we want to read/write mem
);
    
    // 64-item memory arrays: 1-bit valid bit, 24-bit tags, and 32-bit data. 
    logic d_valid [0:63]; 
    logic [23:0] d_tags [0:63]; 
    logic [31:0] d_data [0:63]; 
    // currently read out values of valid-bit, tag, data 
    logic valid = d_valid[index]; 
    logic [23:0] tagout = d_tags[index];
    logic [31:0] c_dout = d_data[index]; 

    // fields of address from cpu 
    logic [23:0] tag = p_a[31:8]; 
    logic [5:0] index = p_a[7:2]; 

    // update the memory. valid bits can be cleared with clrn
    always @(posedge clk or negedge clrn)
        if (!clrn) begin 
            for (integer i = 0; i < 64; i = i + 1)
                d_valid[i] <= 0; 
        end
        else if (c_write) // if writing to cache, valid bit at block index is 1
            d_valid[index] <= 1;
    always @ (posedge clk)
        if (c_write) begin // if writing to cache, write tag and data 
            d_tags[index] <= tag; 
            d_data[index] <= c_din; // data is either from cpu or mem
        end 
    
    // cache data in either from cpu or memory 
    logic [31:0] c_din; 
    assign c_din = p_rw ? p_dout : m_dout;

    // data sent to cpu is either from cache or mem
    assign p_din = cache_hit ? c_dout : m_dout; 

    // forward signals to the memory = write through policy
    assign m_din = p_dout; 
    assign m_a = p_a;    
    assign m_rw = p_rw;
    
    // straight forward
    logic c_write, cache_hit, cache_miss;
    assign cache_hit = p_strobe & valid & (tagout == tag); 
    assign cache_miss = p_strobe & (!valid | (tagout != tag)); 
    assign c_write = ~uncached & (p_rw | cache_miss & m_ready);

    // access memory when writing or cache miss
    assign m_strobe = p_rw | cache_miss; 

    // i guess cpu writing and mem ready = p_ready too
    // (cpu is reading and got cache hit) or (missed and mem is ready)
    assign p_ready = ~p_rw & cache_hit | 
                    (cache_miss | p_rw) & m_ready; 

endmodule

// READING
// basically we send a signal to memory asking for data at addr. 
// we send m_a, m_strobe, m_rw, and receive m_dout, m_ready (m_din sent for writes)
// on the way, we save this data in the cache. 
// if we request the same data again, we get it quicker from the cache. 

// WRITING 
// we write data to both memory and cache. 

// i was expecting some FSM that gets the next N words from memory when we request one word.


module slow_ram(
    input logic clk, clrn, 

    input logic [31:0] m_din, m_a, 
    input logic m_strobe, m_rw,

    output logic [31:0] m_dout, 
    output logic m_ready
);
    logic [31:0] data[0:127];

    // we strobe memory and write to it
    always @ (posedge clk, negedge clrn)
        if (!clrn) begin 
            for (integer i = 0; i < 128; i = i + 1)
                data[i] <= 0;
        end 
        else if (m_strobe & m_rw) data[m_a] <= m_din;

    // the memory is ready after 6 clocks 
    assign m_dout = {32{m_ready}} & data[m_a];
    assign m_ready = (state_reg == ready);



    // counter fsm
    typedef enum {wait1, wait2, wait3, wait4, wait5, ready} state_type;
    state_type state_reg, state_nxt;

    always_ff @(posedge clk, negedge clrn)
        if (!clrn) state_reg <= wait1;
        else state_reg <= state_nxt;

    always_comb 
        case (state_reg)
            wait1 : if (m_strobe & ~m_rw) // if strobe and read, start state transition chain
                        state_nxt = wait2;
                    else 
                        state_nxt = wait1;
            wait2 : state_nxt = wait3;
            wait3 : state_nxt = wait4;
            wait4 : state_nxt = wait5;
            wait5 : state_nxt = ready;
            ready : if (!m_strobe | m_rw) // if not strobe or writing, go to start state
                        state_nxt = wait1;
                    else 
                        state_nxt = ready;
            default: state_nxt = wait1;
        endcase
endmodule

module soy_cache(
    input logic clk, clrn, 

    input logic [31:0] p_dout, p_a, 
    input logic p_strobe, p_rw,

    output logic [31:0] m_din, m_a, 
    output logic m_strobe, m_rw,
    input logic [31:0] m_dout, 
    input logic m_ready

    output logic [31:0] p_din, 
    output logic p_ready
);
    // incremental approach: 
    
    // - request data and get a copy of it from mem, and store it in cache (slow)
    // - request again, it comes (fast)
    // - request different data (slow)
    // - request overwrite of first data (slow)

    // - write to cache hit
    // - request that data, should get new data (fast)
    // - write cache miss: doesnt write to cache, only mem
    // - request that data, should get new data (slow)


    /*
    // first 2 bits of address specify which block, rest is tag
    logic [1:0] index; 
    logic [29:0] tag;
    assign index = p_a[1:0];
    assign tag = p_a[31:2]; 

    // stores 4 words of data
    logic [31:0] data[0:3];
    logic [29:0] tags[0:3];

    always @ (posedge clk, negedge clrn)
        if (!clrn) begin 
            for (integer i = 0; i < 4; i = i + 1) begin 
                data[i] <= 0;
                tags[i] <= 0;
            end
        end 
        else if (p_strobe & p_rw) begin // if cpu writes data, its written to cache too
            data[index] <= c_din;
            tags[index] <= tag;
        end

    // cache data in from cpu or mem
    logic [31:0] c_din; 
    assign c_din = p_rw ? p_dout : m_dout;  

    // cpu signals forwarded to mem because it writes to it too
    assign m_din = p_dout;
    assign m_rw = p_rw;
    assign m_a = p_a;

    // we strobe mem if cpu is writing, or we dont have the data locally 
    assign m_strobe = p_rw | cache_miss;

    // READING 
    logic cache_hit, cache_miss;
    assign cache_hit = p_strobe & (tags[index] == tag);
    assign cache_miss = p_strobe & (tags[index] != tag);



    logic c_write;
    */

endmodule