
module CNNAccelerator(
    input                       clk,
    input                       rst,
    input                       flush,

    // Control cmd_req
    input                       nice_req_valid,
    output                      nice_req_ready,
    input [`NICE_OP_WIDTH-1: 0] nice_req_op,
    input [6: 0]                nice_req_imm,
    input [31: 0]               nice_req_rj,
    input [31: 0]               nice_req_rk,

    // Control cmd_rsp
    output                      nice_rsp_valid,
    input                       nice_rsp_ready,
    output [31: 0]              nice_rsp_rdat,

    // Memory lsu_req
    output                      nice_icb_cmd_valid,
    input                       nice_icb_cmd_ready,
    output [31: 0]              nice_icb_cmd_addr,
    output                      nice_icb_cmd_read,
    output [31: 0]              nice_icb_cmd_wdata,
    output [1: 0]               nice_icb_cmd_size,

    // Memory lsu_rsp
    input                       nice_icb_rsp_valid,
    input [31: 0]               nice_icb_rsp_rdata
);

// decode
    wire op_conf_buf;
    wire op_conf_res;
    wire op_conv;
    assign op_conf_buf  = nice_req_op == 0;
    assign op_conf_res  = nice_req_op == 1;
    assign op_conv      = nice_req_op == 2;

    reg [31: 0] conf_buf_addr;
    reg [`WEIGHT_SIZE-1: 0][31: 0] conf_res_addr;

    always @(posedge clk)begin
        if(nice_req_valid & op_conf_buf)begin
            conf_buf_addr <= nice_req_rj;
        end
    end

// state
    parameter FSM_WIDTH = 2;
    parameter IDLE      = 'b0;
    parameter WEIGHT    = 'b1;
    parameter CONV      = 'b2;
    reg [FSM_WIDTH-1: 0] state_r;
    wire [FSM_WIDTH-1: 0] nxt_state;
    wire state_en;
    wire idle_exit;
    wire weight_exit;
    wire conv_exit;
    wire window_finish;
    wire res_buf_empty;

    wire state_idle     = state_r == IDLE;
    wire state_weight   = state_r == WEIGHT;
    wire state_conv     = state_r == CONV;

    assign state_en = idle_exit | weight_exit | conv_exit;
    assign idle_exit = state_idle & nice_req_valid & op_conv;
    assign weight_exit = state_weight & weight_cmd_end & nice_icb_rsp_valid;
    assign conv_exit = state_conv & window_finish & res_buf_empty;
    assign nxt_state = {FSM_WIDTH{idle_exit}} & WEIGHT |
                       {FSM_WIDTH{weight_exit}} & CONV |
                       {FSM_WIDTH{conv_exit}} & IDLE;

    always_ff @(posedge clk)begin
        if(rst | flush)begin
            state_r <= IDLE;
        end
        else begin
            if(state_en) state_r <= nxt_state;
        end
    end

// weight control
    reg [`WEIGHT_SIZE*`KERNEL_SIZE-1:0][31:0] weight_buf;
    reg [$clog2(`WEIGHT_SIZE*`KERNEL_SIZE)-1:0] weight_idx;
    reg weight_cmd_end;
    reg [31: 0] weight_addr;
    wire weight_cmd_valid;
    wire weight_cmd_hsk = weight_cmd_valid & nice_icb_cmd_ready;

    assign weight_cmd_valid = state_weight & ~weight_cmd_end;

    always @(posedge clk)begin
        if(weight_cmd_hsk)begin
            weight_buf[weight_idx] <= nice_icb_rsp_rdata;
        end
        if(idle_exit)begin
            weight_addr <= nice_req_rj;
        end
        if(weight_cmd_hsk)begin
            weight_addr <= weight_addr + 4;
        end
        if(rst | idle_exit)begin
            weight_cmd_end <= 1'b0;
            weight_idx <= 0;
        end
        else begin
            if(state_weight & weight_cmd_hsk & (weight_idx == `WEIGHT_SIZE*`KERNEL_SIZE - 1)) begin
                weight_cmd_end <= 1'b1;
            end
            if(weight_cmd_hsk)begin
                weight_idx <= weight_idx + 1;
            end
        end
    end

// input buffer
    wire buffer_cmd_valid;
    wire buffer_cmd_ready;
    wire [31: 0] buffer_cmd_addr;
    wire [`KERNEL_SIZE*32-1: 0] window;
    wire window_valid;
    wire window_new;

    CNNBuffer buffer(
        .clk(clk),
        .rst(rst),
        .req(weight_exit),
        .req_addr(conf_buf_addr),
        .req_final(conv_exit),
        .nice_icb_cmd_valid(buffer_cmd_valid),
        .nice_icb_cmd_ready(buffer_cmd_ready),
        .nice_icb_cmd_addr(buffer_cmd_addr),
        .nice_icb_rsp_valid(nice_icb_rsp_valid),
        .nice_icb_rsp_rdata(nice_icb_rsp_rdata),
        .window(window),
        .window_valid(window_valid),
        .window_new(window_new),
        .window_finish(window_finish)
    );

// conv
    wire [`WEIGHT_SIZE-1: 0][31: 0] conv_data;
    wire [`WEIGHT_SIZE-1: 0] conv_valid;
    CNNConv conv(
        .clk(clk),
        .rst(rst),
        .window_valid(window_valid),
        .window_new(window_new),
        .window(window),
        .weight(weight_buf),
        .conv_data(conv_data),
        .conv_valid(conv_valid)
    );

// result buffer
    wire res_cmd_valid;
    wire res_cmd_ready;
    wire res_cmd_hsk;
    wire [31: 0] res_cmd_addr;
    wire [31: 0] res_cmd_wdata;
    reg [$clog2(`WEIGHT_SIZE)-1: 0] res_weight_idx;
    wire [`WEIGHT_SIZE-1: 0] res_weight_vec;
    wire [`WEIGHT_SIZE-1: 0] res_buf_valid;
    wire [`WEIGHT_SIZE-1: 0][31: 0] res_buf_rdata;

    Decoder #(`WEIGHT_SIZE) decoder_res_weight(res_weight_idx, res_weight_vec);
    assign res_cmd_valid = |(res_weight_vec & res_buf_valid);
    assign res_cmd_ready = ~buffer_cmd_valid;
    assign res_cmd_addr = conf_res_addr[res_weight_idx];
    assign res_cmd_wdata = res_buf_rdata[res_weight_idx];
    assign res_buf_empty = ~(|res_buf_valid);

    always @(posedge clk)begin

        if(nice_req_valid & op_conf_res)begin
            conf_res_addr[nice_req_imm[$clog2(`WEIGHT_SIZE)-1:0]] <= nice_req_rj;
        end
        if(res_cmd_hsk)begin
            conf_res_addr[res_weight_idx] <= res_cmd_addr[res_weight_idx] + 4;
        end
        if(rst)begin
            res_weight_idx <= 0;
        end
        else begin
            if(res_cmd_hsk)begin
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
        always_ff @(posedge clk)begin
            if(state_conv & conv_valid[i])begin
                res_buf[i][tail] <= conv_data[i];
            end

            if(rst)begin
                head <= 0;
                tail <= 0;
                hdir <= 1'b0;
                tdir <= 1'b0;
            end
            else begin
                if(state_conv & conv_valid[i])begin
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

    assign nice_req_ready = 1'b1;
    assign nice_rsp_valid = op_conf_buf | conv_exit | op_conf_res;
    assign nice_rsp_rdat  = 0;

    assign nice_icb_cmd_valid = weight_cmd_valid | buffer_cmd_valid | res_cmd_valid;
    assign nice_icb_cmd_addr = weight_cmd_valid ? weight_addr :
                               buffer_cmd_valid ? buffer_cmd_addr : res_cmd_addr;
    assign nice_icb_cmd_read = res_cmd_valid & ~buffer_cmd_valid;
    assign nice_icb_cmd_wdata = res_cmd_wdata;
    assign nice_icb_cmd_size = 2'b10;

endmodule