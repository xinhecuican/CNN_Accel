`include "CNNConfig.vh"


module CNNBuffer(
    input                               clk,
    input                               rst,

    input [`KERNEL_WIDTH-1: 0]          kernel_width_i,
    input [`KERNEL_WIDTH-1: 0]          kernel_height_i,

    input                               req,
    input                               req_final, // op finish

    output                              lacc_data_valid,
    input                               lacc_data_ready,
    input                               lacc_drsp_valid,
    input [31: 0]                       lacc_drsp_rdata,

    output [`WINDOW_SIZE-1: 0][31: 0]   window,
    output                              window_valid,
    output                              window_finish,
    input                               window_stall
);
    reg [$clog2(`BUFFER_WIDTH)-1: 0]    idx_x;
    reg [`BUFFER_DEPTH-1: 0]            vec_y;
    reg                                 last;
    
    reg [`KERNEL_SIZE-1: 0]             kernel_fill;
    reg [$clog2(`BUFFER_WIDTH)-1: 0]    kernel_idx_x;
    reg [$clog2(`BUFFER_DEPTH)-1: 0]    kernel_idx_y;
    reg                                 kernel_data_req_end;
    wire [$clog2(`BUFFER_DEPTH)-1: 0]   idx_y;
    wire                                kernel_data_req;
    wire                                kernel_idx_x_max;
    wire [`KERNEL_SIZE-1: 0][31: 0]     window_rdata;
// state control
    parameter FSM_WIDTH = 1;
    parameter BUFFER_SIZE = `BUFFER_WIDTH * `BUFFER_DEPTH;
    parameter IDLE      = 'b0;
    parameter REQ       = 'b1;
    reg [FSM_WIDTH-1: 0] state_r;
    wire [FSM_WIDTH-1: 0] nxt_state;
    wire state_en;
    wire state_idle = state_r == IDLE;
    wire state_req  = state_r == REQ;
    wire idle_exit  = state_idle & req;
    wire req_exit   = state_req & req_final;

    assign state_en = idle_exit | req_exit;
    assign nxt_state = {FSM_WIDTH{idle_exit}} & REQ |
                       {FSM_WIDTH{req_exit}} & IDLE;    

    always @(posedge clk)begin
        if(rst)begin
            state_r <= IDLE;
        end else begin
            if(state_en) state_r <= nxt_state;
        end
    end

// channel control
    reg                         cmd_valid_r;
    reg                         cmd_read_r;
    wire                        cmd_valid_last;
    reg  [$clog2(BUFFER_SIZE)-1: 0]    req_idx_r;
    wire [$clog2(BUFFER_SIZE)-1: 0]    req_idx_n;
    wire                        req_idx_en;
    wire                        cmd_hsk;

    assign cmd_valid_last   = state_req & lacc_data_ready & (req_idx_r == BUFFER_SIZE - 1);
    assign req_idx_en       = req | cmd_hsk;
    assign cmd_hsk          = lacc_data_ready & cmd_valid_r;
    assign req_idx_n        = idle_exit ? 0 : req_idx_r + 1;

    always @(posedge clk)begin
        if(rst)begin
            cmd_valid_r <= 1'b0;
            req_idx_r <= 0;
        end 
        else begin
            if(idle_exit)begin
                cmd_valid_r <= 1'b1;
            end
            else if(cmd_valid_last) begin
                cmd_valid_r <= 1'b0;
            end
            if(req_idx_en) req_idx_r <= req_idx_n;
        end
    end
    
    assign lacc_data_valid   = cmd_valid_r;

// idx control

    wire idx_x_max = idx_x == `BUFFER_WIDTH - 1;

    always @(posedge clk)begin
        if(rst | req)begin
            idx_x <= 0;
            vec_y <= 1;
            last  <= 1'b0;
        end
        else begin
            if(lacc_drsp_valid)begin
                idx_x <= idx_x_max ? 0 : idx_x + 1;
                if(idx_x_max) begin
                    vec_y <= {vec_y[`BUFFER_DEPTH-2: 0], vec_y[`BUFFER_DEPTH-1]};
                    last <= vec_y[`BUFFER_DEPTH-1];
                end
            end
        end
    end

// buffer control
    wire [`BUFFER_DEPTH-1: 0][31: 0] rdata;
    genvar i, j;
generate
    for(i=0; i<`BUFFER_DEPTH; i=i+1)begin : gen_buffer
        reg [31: 0] buffer_row [`BUFFER_WIDTH-1: 0];
        assign rdata[i] = buffer_row[kernel_idx_x];
        always_ff @(posedge clk)begin
            if(lacc_drsp_valid & vec_y[i])begin
                buffer_row[idx_x] <= lacc_drsp_rdata;
            end
        end
    end
endgenerate

// window gen
    Encoder #(`BUFFER_DEPTH) encoder_idx_y (vec_y, idx_y);

    assign kernel_data_req = state_req & ~kernel_data_req_end & ~window_stall &
                            ((idx_y > kernel_idx_y) | last | (idx_y == kernel_idx_y) & (kernel_idx_x < idx_x));
    assign kernel_idx_x_max = kernel_idx_x == `BUFFER_WIDTH - 1;
    always @(posedge clk)begin
        if(rst | req)begin
            kernel_fill <= 0;
            kernel_idx_x <= 0;
            kernel_idx_y <= kernel_height_i - 1;
            kernel_data_req_end <= 1'b0;
        end 
        else begin
            if(kernel_data_req)begin
                kernel_idx_x <= kernel_idx_x_max ? 0 : kernel_idx_x + 1;
                kernel_fill <= ((kernel_fill << 1) | 1'b1) & {`KERNEL_SIZE{~kernel_idx_x_max}};
                if(kernel_idx_x_max)begin
                    kernel_idx_y <= kernel_idx_y + 1;
                    if(kernel_idx_y == `BUFFER_DEPTH - 1)begin
                        kernel_data_req_end <= 1'b1;
                    end
                end
            end
        end
    end

generate
    for(i=0; i<`KERNEL_SIZE; i=i+1)begin : gen_window_rdata
        wire [$clog2(`BUFFER_DEPTH)-1: 0] window_ridx;
        assign window_ridx = kernel_idx_y - kernel_height_i + 1 + i;
        assign window_rdata[i] = rdata[window_ridx];
    end
endgenerate


    reg [`WINDOW_SIZE-1: 0][31: 0]  window_r;
    reg                             window_valid_r;
    reg [$clog2(`KERNEL_SIZE)-1: 0] window_col_idx;
    wire [`KERNEL_SIZE-1: 0]        window_col;

    Decoder #(`KERNEL_SIZE) decoder_window_col(window_col_idx, window_col);

    wire kernel_fill_all = &kernel_fill;
generate
    for(i=0; i<`KERNEL_SIZE; i=i+1)begin : gen_window
        for(j=0; j<`KERNEL_SIZE; j=j+1)begin : gen_window_row
            wire window_fill_en = kernel_data_req & (kernel_fill_all | ~kernel_fill_all & window_col[j]);
            wire [31: 0] window_wdata;
            assign window_wdata = (j + 1 != kernel_height_i) & kernel_fill_all ? 
                                    window_r[i*`KERNEL_SIZE+j+1] : window_rdata[i];
            always @(posedge clk)begin
                if(window_fill_en) window_r[i * `KERNEL_SIZE + j] <= window_wdata;
            end
        end
    end
endgenerate

    wire [`KERNEL_WIDTH-1: 0] kernel_fill_idx;
    assign kernel_fill_idx = kernel_width_i > 1 ? kernel_width_i - 2 : 0;
    always @(posedge clk)begin
        if(~window_stall)begin
            window_valid_r <= kernel_data_req & kernel_fill[kernel_fill_idx];
        end
        if(rst)begin
            window_col_idx <= 0;
        end
        else begin 
            if(kernel_data_req)begin
                window_col_idx <= kernel_idx_x_max | (window_col_idx == kernel_width_i - 1) ? 0 : window_col_idx + 1;
            end
        end
    end
    assign window           = window_r;
    assign window_valid     = window_valid_r;
    assign window_finish    = kernel_data_req_end;
endmodule