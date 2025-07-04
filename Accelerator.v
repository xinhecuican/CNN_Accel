`include "CNNConfig.vh"
`include "mycpu.h"

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
    reg [`KERNEL_WIDTH-1: 0]            conf_kernel_width;
    reg [`KERNEL_WIDTH-1: 0]            conf_kernel_height;
    reg [$clog2(`BUFFER_DEPTH)-1: 0]    conf_buf_depth;
    reg [$clog2(`BUFFER_WIDTH)-1: 0]    conf_buf_width;
    reg [`STRIDE_WIDTH-1: 0]            conf_stride;
    reg [$clog2(`WEIGHT_SIZE)-1: 0]     conf_weight_num;
    reg [`KERNEL_WIDTH-1: 0]            conf_padding;
    reg [3: 0]                          conf_padding_valid;
    reg [1: 0]                          conf_weight_size;
    reg [1: 0]                          conf_buf_size;
    reg                                 conf_add_write;
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
    reg [`WEIGHT_SIZE*32-1: 0] conf_res_addr;

    always @(posedge clk)begin
        conf_refresh <= lacc_req_valid & op_conf_buf;
        if(lacc_req_valid & op_conf_buf)begin
            conf_buf_addr <= lacc_req_rj;
        end
        else if(buffer_cmd_valid & buffer_cmd_ready)begin
            conf_buf_addr <= conf_buf_addr + 4;
        end
        if(lacc_req_valid & op_conf_buf)begin
            conf_kernel_width   <= lacc_req_rk[0 +: `KERNEL_WIDTH];
            conf_kernel_height  <= lacc_req_rk[3 +: `KERNEL_WIDTH];
            conf_buf_width      <= lacc_req_rk[6 +: $clog2(`BUFFER_WIDTH)];
            conf_buf_depth      <= lacc_req_rk[11 +: $clog2(`BUFFER_DEPTH)];
            conf_stride         <= lacc_req_rk[16 +: `STRIDE_WIDTH];
            conf_weight_num     <= lacc_req_rk[18 +: $clog2(`WEIGHT_SIZE)];
            conf_padding_valid  <= lacc_req_rk[21 +: 4];
            conf_padding        <= lacc_req_rk[25 +: `KERNEL_WIDTH];
            conf_add_write      <= lacc_req_rk[27];
            conf_weight_size    <= lacc_req_rk[28 +: 2];
            conf_buf_size       <= lacc_req_rk[30 +: 2];
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
    reg [`WEIGHT_SIZE*`WINDOW_SIZE*32-1:0] weight_buf;
    reg [$clog2(`WEIGHT_SIZE)-1: 0] weight_size, req_weight_size;
    reg [`KERNEL_WIDTH-1: 0] weight_row_idx, weight_col_idx;
    reg [`KERNEL_WIDTH-1: 0] req_weight_row_idx, req_weight_col_idx;
    reg [$clog2(`WEIGHT_SIZE*`WINDOW_SIZE)-1:0] weight_idx, weight_line_idx, weight_page_idx;
    wire [$clog2(`WEIGHT_SIZE*`WINDOW_SIZE)-1:0] weight_line_idx_n, weight_page_idx_n;
    reg weight_cmd_end;
    reg [31: 0] weight_addr;
    reg [1: 0] weight_offset;
    wire [2: 0] weight_addr_cin;
    wire [31: 0] weight_data_in;
    wire weight_addr_en;
    wire [31: 0] nxt_weight_addr;
    wire weight_cmd_valid;
    wire weight_data_valid;
    wire weight_cmd_hsk = weight_cmd_valid & lacc_data_ready;

    assign weight_cmd_valid = state_weight & ~weight_cmd_end;
    assign weight_data_valid = state_weight & lacc_drsp_valid;
    assign weight_addr_en = idle_exit | weight_cmd_hsk;
    assign weight_addr_cin = {conf_weight_size[1], conf_weight_size[0], ~(|conf_weight_size)};
    assign nxt_weight_addr = {32{idle_exit}} & lacc_req_rj |
                             {32{weight_cmd_hsk}} & (weight_addr + weight_addr_cin);

    wire req_weight_size_end    = req_weight_size == conf_weight_num;
    wire req_weight_row_end     = req_weight_row_idx == conf_kernel_height - 1;
    wire req_weight_col_end     = req_weight_col_idx == conf_kernel_width - 1;
    wire weight_size_end        = weight_size == conf_weight_num;
    wire weight_row_end         = weight_row_idx == conf_kernel_height - 1;
    wire weight_col_end         = weight_col_idx == conf_kernel_width - 1;

    assign weight_line_idx_n    = weight_line_idx + `KERNEL_SIZE;
    assign weight_page_idx_n    = weight_page_idx + `WINDOW_SIZE;

    RDataGen weight_data_gen (conf_weight_size, weight_offset, lacc_drsp_rdata, weight_data_in);

    always @(posedge clk)begin
        if(weight_data_valid)begin
            weight_buf[weight_idx*32 +: 32] <= weight_data_in;
        end
        if(weight_addr_en) weight_addr <= nxt_weight_addr;
        if(rst | idle_exit)begin
            weight_cmd_end <= 1'b0;
            weight_idx <= 0;
            weight_size <= 0;
            req_weight_size <= 0;
            weight_row_idx <= 0;
            weight_col_idx <= 0;
            req_weight_col_idx <= 0;
            req_weight_row_idx <= 0;
            weight_line_idx <= 0;
            weight_page_idx <= 0;
            weight_offset <= 0;
        end
        else begin
            if(state_weight & weight_cmd_hsk & 
                req_weight_col_end & req_weight_row_end & req_weight_size_end) begin
                weight_cmd_end <= 1'b1;
            end
            if(weight_cmd_hsk)begin
                req_weight_col_idx <= req_weight_col_end ? 0 : req_weight_col_idx + 1;
                if(req_weight_col_end)begin
                    req_weight_row_idx <= req_weight_row_end ? 0 : req_weight_row_idx + 1;
                    if(req_weight_row_end)begin
                        req_weight_size <= req_weight_size + 1;
                    end
                end

            end
            if(weight_data_valid)begin
                weight_offset <= weight_offset + weight_addr_cin;
                weight_idx <= weight_col_end & weight_row_end ? weight_page_idx_n :
                              weight_col_end ? weight_line_idx_n : weight_idx + 1;
                weight_col_idx <= weight_col_end ? 0 : weight_col_idx + 1;
                if(weight_col_end)begin
                    weight_row_idx <= weight_row_end ? 0 : weight_row_idx + 1;
                    weight_line_idx <= weight_row_end ? weight_page_idx_n : weight_line_idx_n;
                    if(weight_row_end)begin
                        weight_size <= weight_size + 1;
                        weight_page_idx <= weight_page_idx_n;
                    end
                end
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
        .stride_i(conf_stride),
        .padding_i(conf_padding),
        .padding_valid_i(conf_padding_valid),
        .buf_size_i(conf_buf_size),
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
    reg act_op;
    wire [`WEIGHT_SIZE*32-1: 0] conv_data;
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
        .act_valid(act_op),
        .conf_refresh(conf_refresh),
        .kernel_height(kernel_height_vec[`KERNEL_SIZE: 1]),
        .kernel_width(kernel_width_vec[`KERNEL_SIZE: 1]),
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
            act_op  <= 0;
        end
        else if(lacc_req_valid & op_conv)begin
            conv_op <= lacc_req_imm[0];
            pool_op <= lacc_req_imm[1];
            act_op  <= lacc_req_imm[2];
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
    wire [`WEIGHT_SIZE*32-1: 0] res_buf_rdata;
    wire [31: 0]                res_buf_rdata_sel;
    wire [`WEIGHT_SIZE*32-1: 0] res_buf_wdata;
    wire [`WEIGHT_SIZE-1: 0] res_buf_en;
    reg  [`WEIGHT_SIZE-1: 0] conf_res_buf_valid;
    reg          res_buf_read;
    reg          res_buf_wait;
    reg  [31: 0] res_drsp_rdata;
    wire [31: 0] nxt_res_drsp_rdata;


    Decoder #(`WEIGHT_SIZE) decoder_res_weight(res_weight_idx, res_weight_vec);
    assign res_cmd_valid    = (|(res_weight_vec & res_buf_valid)) & ~res_buf_wait;
    assign res_cmd_ready    = ~buffer_cmd_valid & lacc_data_ready;
    assign res_cmd_hsk      = res_cmd_valid & res_cmd_ready;
    assign res_cmd_addr     = conf_res_addr[res_weight_idx*32 +: 32];
    assign res_buf_rdata_sel = res_buf_rdata[res_weight_idx*32 +: 32];
    assign res_cmd_wdata    = conf_add_write ? res_drsp_rdata : res_buf_rdata_sel;
    assign res_buf_empty    = ~(|res_buf_valid);
    assign res_buf_stall    = |res_buf_full;
    assign res_cmd_addr_n4  = res_cmd_addr + 4;
    assign res_buf_en       = conf_res_buf_valid & (conv_valid | pool_valid);

    assign res_addr_widx = {WEIGHT_WIDTH{lacc_req_valid & op_conf_res}} & lacc_req_rk[WEIGHT_WIDTH-1:0] |
                           {WEIGHT_WIDTH{res_cmd_hsk & ~res_buf_read}} & res_weight_idx;
    assign res_addr_we   = lacc_req_valid & op_conf_res | res_cmd_hsk & ~res_buf_read;
    assign res_addr_wdata = {32{lacc_req_valid & op_conf_res}} & lacc_req_rj |
                            {32{res_cmd_hsk}} & res_cmd_addr_n4;  
    assign res_weight_en = res_cmd_hsk & conv_op & ~res_buf_read;

    wire res_drsp_rdata_en = res_buf_wait & lacc_drsp_valid;
    assign nxt_res_drsp_rdata = lacc_drsp_rdata + res_buf_rdata_sel;

    always @(posedge clk)begin
        if(conf_refresh) conf_res_buf_valid <= (1 << (conf_weight_num+1)) - 1;
        if(res_addr_we) conf_res_addr[res_addr_widx*32 +: 32] <= res_addr_wdata;
        if(res_drsp_rdata_en) res_drsp_rdata <= nxt_res_drsp_rdata;
        if(rst | idle_exit)begin
            res_weight_idx <= 0;
            res_buf_read <= conf_add_write;
            res_buf_wait <= 0;
        end
        else begin
            if(res_cmd_valid & res_cmd_ready & conf_add_write)begin
                res_buf_read <= ~res_buf_read;
            end
            if(res_cmd_hsk & res_buf_read)begin
                res_buf_wait <= 1'b1;
            end
            if(res_buf_wait & lacc_drsp_valid)begin
                res_buf_wait <= 1'b0;
            end
            if(res_weight_en)begin
                res_weight_idx <= res_weight_idx == conf_weight_num ? 0 : res_weight_idx + 1;
            end
        end
    end

    genvar i;
generate
    for(genvar i=0; i<`WEIGHT_SIZE; i=i+1)begin : gen_res_buf
        reg [`RES_BUF_SIZE*32-1: 0] res_buf;
        reg [$clog2(`RES_BUF_SIZE)-1: 0] head, tail;
        reg hdir, tdir;
        wire equal = head == tail;
        wire dir_xor = hdir ^ tdir;
        wire full = equal & dir_xor;
        wire empty = equal & ~dir_xor;
        assign res_buf_valid[i] = ~empty;
        assign res_buf_rdata[i*32 +: 32] = res_buf[head*32 +: 32];
        assign res_buf_full[i]  = full;
        if(i == 0)begin
            assign res_buf_wdata[i*32 +: 32] = {32{conv_valid[i]}} & conv_data[i*32 +: 32] |
                                      {32{pool_valid}} & pool_data;
        end
        else begin
            assign res_buf_wdata[i*32 +: 32] = conv_data[i*32 +: 32];
        end
        always @(posedge clk)begin
            if(state_conv & res_buf_en[i] & ~res_buf_stall)begin
                res_buf[tail*32 +: 32] <= res_buf_wdata[i*32 +: 32];
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
                if(res_cmd_hsk & res_weight_vec[i] & ~res_buf_read)begin
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
    assign lacc_data_read = ~(res_cmd_valid & ~buffer_cmd_valid & ~res_buf_read);
    assign lacc_data_wdata = res_cmd_wdata;
    assign lacc_data_size = 2'b10;

endmodule