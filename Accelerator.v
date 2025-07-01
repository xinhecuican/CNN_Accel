`include "CNNConfig.vh"

module CNNAccelerator(
    input                       clk,
    input                       rst,


    input                       lacc_flush,

    input                       lacc_req_valid,
    input [`LACC_OP_WIDTH-1: 0] lacc_req_op,
    input [6: 0]                lacc_req_imm,
    input [31: 0]               lacc_req_rj,
    input [31: 0]               lacc_req_rk,

    output                      lacc_rsp_valid,
    output [31: 0]              lacc_rsp_rdat,

    // wreq will also send valid sign
    output                      lacc_data_valid,
    input                       lacc_data_ready,
    output [31: 0]              lacc_data_addr,
    output                      lacc_data_read,
    output [31: 0]              lacc_data_wdata,
    output [1: 0]               lacc_data_size,

    input                       lacc_drsp_valid,
    input [31: 0]               lacc_drsp_rdata
);
    wire buffer_cmd_valid;
    wire buffer_cmd_ready;
    wire buffer_drsp_valid;
    wire buffer_req;
    wire [31: 0] buffer_cmd_addr;
    wire [`WINDOW_SIZE*32-1: 0] window;
    wire window_valid;
    reg data_req_buf;
    reg [`KERNEL_WIDTH-1: 0] conf_kernel_width;
    reg [`KERNEL_WIDTH-1: 0] conf_kernel_height;
    reg [$clog2(`BUFFER_DEPTH)-1: 0] conf_buf_depth;
    reg [$clog2(`BUFFER_WIDTH)-1: 0] conf_buf_width;
    wire [`KERNEL_SIZE: 0] kernel_width_vec, kernel_height_vec;
    reg conf_refresh;
// decode
    wire op_conf_buf;
    wire op_conf_res;
    wire op_conv;
    assign op_conf_buf  = lacc_req_op == 0;
    assign op_conf_res  = lacc_req_op == 1;
    assign op_conv      = lacc_req_op == 2;

    reg [31: 0] conf_buf_addr;
    reg [`WEIGHT_SIZE-1: 0][31: 0] conf_res_addr;

    always @(posedge clk)begin
        conf_refresh <= lacc_req_valid & op_conf_buf;
        if(lacc_req_valid & op_conf_buf)begin
            conf_buf_addr <= lacc_req_rj;
        end
        else if(buffer_cmd_valid & buffer_cmd_ready)begin
            conf_buf_addr <= conf_buf_addr + 4;
        end
        if(lacc_req_valid & op_conf_buf)begin
            conf_kernel_width <= lacc_req_rk[0 +: `KERNEL_WIDTH];
            conf_kernel_height <= lacc_req_rk[4 +: `KERNEL_WIDTH];
            conf_buf_width <= lacc_req_rk[8 +: $clog2(`BUFFER_WIDTH)];
            conf_buf_depth <= lacc_req_rk[14 +: $clog2(`BUFFER_DEPTH)];
        end
    end
    Decoder #(`KERNEL_SIZE+1) decoder_kernel_width(conf_kernel_width, kernel_width_vec);
    Decoder #(`KERNEL_SIZE+1) decoder_kernel_height(conf_kernel_height, kernel_height_vec);

// state
    parameter FSM_WIDTH = 2;
    parameter IDLE      = 'b0;
    parameter WEIGHT    = 'b1;
    parameter CONV      = 'd2;
    reg [FSM_WIDTH-1: 0] state_r;
    wire [FSM_WIDTH-1: 0] nxt_state;
    wire [FSM_WIDTH-1: 0] idle_state_n;
    wire state_en;
    wire idle_exit;
    wire weight_exit;
    wire conv_exit;
    wire window_finish;
    wire window_stall;
    wire res_buf_empty;

    wire state_idle     = state_r == IDLE;
    wire state_weight   = state_r == WEIGHT;
    wire state_conv     = state_r == CONV;

    assign state_en = idle_exit | weight_exit | conv_exit;
    assign idle_exit = state_idle & lacc_req_valid & op_conv;
    assign weight_exit = state_weight & weight_cmd_end & lacc_drsp_valid;
    assign conv_exit = state_conv & window_finish & res_buf_empty;
    assign idle_state_n = lacc_req_imm[0] ? WEIGHT : CONV;
    assign nxt_state = {FSM_WIDTH{idle_exit}} & idle_state_n |
                       {FSM_WIDTH{weight_exit}} & CONV |
                       {FSM_WIDTH{conv_exit}} & IDLE;

    always @(posedge clk)begin
        if(rst | lacc_flush)begin
            state_r <= IDLE;
        end
        else begin
            if(state_en) state_r <= nxt_state;
        end
    end

// weight control
    reg [`WEIGHT_SIZE*`WINDOW_SIZE-1:0][31:0] weight_buf;
    reg [$clog2(`WEIGHT_SIZE*`WINDOW_SIZE)-1:0] weight_idx, weight_req_idx;
    reg weight_cmd_end;
    reg [31: 0] weight_addr;
    wire weight_cmd_valid;
    wire weight_data_valid;
    wire weight_cmd_hsk = weight_cmd_valid & lacc_data_ready;

    assign weight_cmd_valid = state_weight & ~weight_cmd_end;
    assign weight_data_valid = state_weight & lacc_drsp_valid;

    always @(posedge clk)begin
        if(weight_data_valid)begin
            weight_buf[weight_idx] <= lacc_drsp_rdata;
        end
        if(idle_exit)begin
            weight_addr <= lacc_req_rj;
        end
        if(weight_cmd_hsk)begin
            weight_addr <= weight_addr + 4;
        end
        if(rst | idle_exit)begin
            weight_cmd_end <= 1'b0;
            weight_idx <= 0;
            weight_req_idx <= 0;
        end
        else begin
            if(state_weight & weight_cmd_hsk & (weight_req_idx == `WEIGHT_SIZE*`WINDOW_SIZE - 1)) begin
                weight_cmd_end <= 1'b1;
            end
            if(weight_cmd_hsk)begin
                weight_req_idx <= weight_req_idx + 1;
            end
            if(weight_data_valid)begin
                weight_idx <= weight_idx + 1;
            end
        end
    end

// input buffer

    assign buffer_cmd_addr = conf_buf_addr;
    assign buffer_cmd_ready = lacc_data_ready;
    assign buffer_drsp_valid = lacc_drsp_valid & data_req_buf;
    assign buffer_req = weight_exit | idle_exit & ~lacc_req_imm[0];
    CNNBuffer buffer(
        .clk(clk),
        .rst(rst),
        .kernel_width_i(conf_kernel_width),
        .kernel_height_i(conf_kernel_height),
        .buffer_width_i(conf_buf_width),
        .buffer_depth_i(conf_buf_depth),
        .req(buffer_req),
        .req_final(conv_exit),
        .lacc_data_valid(buffer_cmd_valid),
        .lacc_data_ready(buffer_cmd_ready),
        .lacc_drsp_valid(buffer_drsp_valid),
        .lacc_drsp_rdata(lacc_drsp_rdata),
        .window(window),
        .window_valid(window_valid),
        .window_finish(window_finish),
        .window_stall(window_stall)
    );

// conv
    reg conv_op;
    reg pool_op;
    wire [`WEIGHT_SIZE-1: 0][31: 0] conv_data;
    wire [`WEIGHT_SIZE-1: 0] conv_valid;
    wire res_buf_stall;
    wire [31: 0] pool_data;
    wire pool_valid;
    wire conv_window_valid;
    wire pool_window_valid;
    wire window_stall_conv;
    wire window_stall_pool;
    reg [`POOL_MODE_WIDTH-1: 0] pool_mode;

    assign conv_window_valid = window_valid & conv_op;
    CNNConv conv(
        .clk(clk),
        .rst(rst),
        .window_valid(conv_window_valid),
        .window(window),
        .weight(weight_buf),
        .conv_data(conv_data),
        .conv_valid(conv_valid),
        .window_stall(window_stall_conv),
        .stall(res_buf_stall)
    );

    assign pool_window_valid = window_valid & pool_op;
    CNNPool pool(
        .clk(clk),
        .rst(rst),
        .conf_refresh(conf_refresh),
        .kernel_height(kernel_height_vec[`KERNEL_SIZE: 1]),
        .kernel_width(kernel_width_vec[`KERNEL_SIZE: 1]),
        .stall(res_buf_stall),
        .pool_mode(pool_mode),
        .window_valid(pool_window_valid),
        .window(window),
        .window_stall(window_stall_pool),
        .pool_data(pool_data),
        .pool_valid(pool_valid)
    );

    assign window_stall = window_stall_conv | window_stall_pool;

    always @(posedge clk)begin
        if(rst)begin
            conv_op <= 0;
            pool_op <= 0;
        end
        else if(lacc_req_valid & op_conv)begin
            conv_op <= lacc_req_imm[0];
            pool_op <= lacc_req_imm[1];
        end
    end

// result buffer
    parameter WEIGHT_WIDTH = $clog2(`WEIGHT_SIZE);
    wire res_cmd_valid;
    wire res_cmd_ready;
    wire res_cmd_hsk;
    wire res_addr_we;
    wire res_wieght_en;
    wire [WEIGHT_WIDTH-1: 0] res_addr_widx;
    wire [31: 0] res_addr_wdata;
    wire [31: 0] res_cmd_addr;
    wire [31: 0] res_cmd_addr_n4;
    wire [31: 0] res_cmd_wdata;
    reg [WEIGHT_WIDTH-1: 0] res_weight_idx;
    wire [`WEIGHT_SIZE-1: 0] res_weight_vec;
    wire [`WEIGHT_SIZE-1: 0] res_buf_valid;
    wire [`WEIGHT_SIZE-1: 0] res_buf_full;
    wire [`WEIGHT_SIZE-1: 0][31: 0] res_buf_rdata;
    wire [`WEIGHT_SIZE-1: 0][31: 0] res_buf_wdata;
    wire [`WEIGHT_SIZE-1: 0] res_buf_en;

    Decoder #(`WEIGHT_SIZE) decoder_res_weight(res_weight_idx, res_weight_vec);
    assign res_cmd_valid    = |(res_weight_vec & res_buf_valid);
    assign res_cmd_ready    = ~buffer_cmd_valid & lacc_data_ready;
    assign res_cmd_hsk      = res_cmd_valid & res_cmd_ready;
    assign res_cmd_addr     = conf_res_addr[res_weight_idx];
    assign res_cmd_wdata    = res_buf_rdata[res_weight_idx];
    assign res_buf_empty    = ~(|res_buf_valid);
    assign res_buf_stall    = |res_buf_full;
    assign res_cmd_addr_n4  = res_cmd_addr + 4;
    assign res_buf_en       = conv_valid | pool_valid;

    assign res_addr_widx = {WEIGHT_WIDTH{lacc_req_valid & op_conf_res}} & lacc_req_rk[WEIGHT_WIDTH-1:0] |
                           {WEIGHT_WIDTH{res_cmd_hsk}} & res_weight_idx;
    assign res_addr_we   = lacc_req_valid & op_conf_res | res_cmd_hsk;
    assign res_addr_wdata = {32{lacc_req_valid & op_conf_res}} & lacc_req_rj |
                            {32{res_cmd_hsk}} & res_cmd_addr_n4;  
    assign res_weight_en = res_cmd_hsk & conv_op;
    always @(posedge clk)begin
        if(res_addr_we) conf_res_addr[res_addr_widx] <= res_addr_wdata;
        if(rst | idle_exit)begin
            res_weight_idx <= 0;
        end
        else begin
            if(res_weight_en)begin
                res_weight_idx <= res_weight_idx == `WEIGHT_SIZE - 1 ? 0 : res_weight_idx + 1;
            end
        end
    end

    genvar i;
generate
    for(genvar i=0; i<`WEIGHT_SIZE; i=i+1)begin : gen_res_buf
        reg [`RES_BUF_SIZE-1: 0][31: 0] res_buf;
        reg [$clog2(`RES_BUF_SIZE)-1: 0] head, tail;
        reg hdir, tdir;
        wire equal = head == tail;
        wire dir_xor = hdir ^ tdir;
        wire full = equal & dir_xor;
        wire empty = equal & ~dir_xor;
        assign res_buf_valid[i] = ~empty;
        assign res_buf_rdata[i] = res_buf[head];
        assign res_buf_full[i]  = full;
        if(i == 0)begin
            assign res_buf_wdata[i] = {32{conv_valid[i]}} & conv_data[i] |
                                      {32{pool_valid}} & pool_data;
        end
        else begin
            assign res_buf_wdata[i] = conv_data[i];
        end
        always @(posedge clk)begin
            if(state_conv & res_buf_en[i] & ~res_buf_stall)begin
                res_buf[tail] <= res_buf_wdata[i];
            end

            if(rst)begin
                head <= 0;
                tail <= 0;
                hdir <= 1'b0;
                tdir <= 1'b0;
            end
            else begin
                if(state_conv & res_buf_en[i] & ~res_buf_stall)begin
                    tail <= tail + 1;
                    if(&tail)begin
                        tdir <= ~tdir;
                    end
                end
                if(res_cmd_hsk & res_weight_vec[i])begin
                    head <= head + 1;
                    if(&head)begin
                        hdir <= ~hdir;
                    end
                end
            end
        end
    end
endgenerate

    always @(posedge clk)begin
        data_req_buf <= buffer_cmd_valid;
    end
    assign lacc_rsp_valid = op_conf_buf | conv_exit | op_conf_res;
    assign lacc_rsp_rdat  = 0;

    assign lacc_data_valid = weight_cmd_valid | buffer_cmd_valid | res_cmd_valid;
    assign lacc_data_addr = weight_cmd_valid ? weight_addr :
                            buffer_cmd_valid ? buffer_cmd_addr : res_cmd_addr;
    assign lacc_data_read = ~(res_cmd_valid & ~buffer_cmd_valid);
    assign lacc_data_wdata = res_cmd_wdata;
    assign lacc_data_size = 2'b10;

endmodule