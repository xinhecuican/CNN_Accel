`include "CNNConfig.vh"

module CNNConv (
    input clk,
    input rst,
    input stall,
    input [`WEIGHT_SIZE*`WINDOW_SIZE*32-1: 0] weight,
    input                               act_valid,
    input [`WEIGHT_SIZE*32-1: 0]        bias,
    input                               conf_refresh,
    input [`KERNEL_SIZE-1: 0]           kernel_height,
    input [`KERNEL_SIZE-1: 0]           kernel_width,

    input                               window_valid,
    input [`WINDOW_SIZE*32-1: 0]        window,
    output                              window_stall,
    output [`WEIGHT_SIZE*32-1: 0]       conv_data,
    output [`WEIGHT_SIZE-1: 0]          conv_valid,
    output                              conv_empty
);

    reg [`WINDOW_SIZE+`WEIGHT_SIZE-1: 0]    conv_valid_r;
    reg [`WINDOW_SIZE*`WEIGHT_SIZE*32-1: 0] res_r;
    wire [`WEIGHT_SIZE*32-1: 0]             res, relu_o;
    parameter CVW = $clog2(`WINDOW_SIZE+`WEIGHT_SIZE);
    reg [CVW-1: 0] conv_valid_idx;
    wire [CVW-1: 0] conv_valid_idx_n;

    wire stall_all      = (|conv_valid) & stall;
    assign window_stall = stall_all;

    assign conv_valid_idx_n = {CVW{kernel_height[1] & kernel_width[1]}} & 'd4 |
                              {CVW{kernel_height[2] & kernel_width[2]}} & 'd8;

    always @(posedge clk)begin
        if(rst)begin
            conv_valid_r <= 0;
            conv_valid_idx <= 0;
        end
        else begin
            if(conf_refresh)begin
                conv_valid_idx <= conv_valid_idx_n;
            end
            if(!stall_all)begin
                conv_valid_r <= {conv_valid_r[`WINDOW_SIZE+`WEIGHT_SIZE-2: 0], window_valid};
            end
        end
    end

    assign conv_data = act_valid ? relu_o : res;
    assign conv_empty = ~(|conv_valid_r);

    genvar i, j;
generate
    for(i=0; i<`WEIGHT_SIZE; i=i+1)begin
        assign conv_valid[i] = conv_valid_r[conv_valid_idx+1+i];
        assign res[i*32 +: 32] = $signed(res_r[(conv_valid_idx*`WEIGHT_SIZE+i)*32 +: 32]) + $signed(bias[i*32 +: 32]);
        Relu relu(res[i*32 +: 32], relu_o[i*32 +: 32]);
    end
    for(i=0; i<`WINDOW_SIZE; i=i+1)begin
        reg [(i+1)*32-1: 0] input_buffer;
        always @(posedge clk)begin
            if(~stall_all)begin
                if(window_valid)begin
                    input_buffer[i*32 +: 32] <= window[i*32 +: 32];
                end
                else begin
                    input_buffer[i*32 +: 32] <= 0;
                end
            end
        end
        for(j=i-1; j>=0; j=j-1)begin
            always @(posedge clk)begin
                if(~stall_all) input_buffer[j*32 +: 32] <= input_buffer[(j+1)*32 +: 32];
            end
        end

        reg [(`WEIGHT_SIZE-1)*32-1: 0] d_r;
        wire [`WEIGHT_SIZE*32-1: 0] d_o;
        always @(posedge clk)begin
            if(~stall_all) d_r <= d_o[(`WEIGHT_SIZE-1)*32-1: 0];
        end
        for(j=0; j<`WEIGHT_SIZE; j=j+1)begin
            wire [31: 0] d_conv, d_i;
            if(i == 0)begin
                assign d_conv = 0;
            end
            else begin
                assign d_conv = res_r[((i-1)*`WEIGHT_SIZE+j)*32 +: 32];
            end
            if(j == 0)begin
                assign d_i = input_buffer[31: 0];
            end
            else begin
                assign d_i = d_r[(j-1)*32 +: 32];
            end

            wire [31: 0] res;
            ConvPE #(i) pe (
                .d_conv(d_conv),
                .d(d_i),
                .w(weight[(j*`WINDOW_SIZE+i)*32 +: 32]),
                .d_o(d_o[j*32 +: 32]),
                .res(res)
            );
            always @(posedge clk)begin
                if(~stall_all) res_r[(i*`WEIGHT_SIZE+j)*32 +: 32] <= res;
            end
        end
    end
endgenerate

endmodule

module ConvPE #(
    parameter WITH_ADD=1
)(
    input [31: 0] d_conv,
    input [31: 0] d,
    input [31: 0] w,
    output [31: 0] d_o,
    output [31: 0] res
);
    wire signed [31: 0] mul_out;
    assign mul_out = $signed(d) * $signed(w);

generate
    if(WITH_ADD)begin
        assign res = $signed(d_conv) + $signed(mul_out);
    end
    else begin
        assign res = mul_out;
    end
endgenerate
    assign d_o = d;
endmodule