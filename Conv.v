

module CNNConv (
    input clk,
    input rst,

    input window_valid,
    input window_new,
    input [`KERNEL_SIZE-1: 0][31: 0] window,
    input [`WEIGHT_SIZE-1: 0][`KERNEL_SIZE-1: 0][31: 0] weight,
    output [`WEIGHT_SIZE-1: 0][31: 0] conv_data,
    output [`WEIGHT_SIZE-1: 0] conv_valid
);


    genvar i;
generate
    for(i=0; i<`WEIGHT_SIZE; i=i+1)begin : gen_conv
        ConvUnit conv_unit (
            .clk(clk),
            .rst(rst),
            .window_valid(window_valid),
            .window_new(window_new),
            .window(window),
            .weight(weight[i]),
            .conv_data(conv_data[i]),
            .conv_valid(conv_valid[i])
        );
    end
endgenerate

endmodule

module ConvUnit(
    input clk,
    input rst,

    input window_valid,
    input window_new,
    input [`KERNEL_SIZE-1: 0][31: 0] window,
    input [`KERNEL_SIZE-1: 0][31: 0] weight,

    output [31: 0] conv_data,
    output conv_valid
);
// level1: mul
wire signed [`KERNEL_SIZE-1: 0][31: 0] mul_out;
genvar i, j;
generate
for (i = 0; i < `KERNEL_SIZE; i = i + 1) begin : gen_mul
    assign mul_out[i] = $signed(window[i]) * $signed(weight[i]);
end
endgenerate

reg signed [`KERNEL_SIZE-1: 0][31: 0] mul_out_r;
reg mul_valid_r;
always_ff @(posedge clk)begin
if(window_valid)begin
    mul_out_r <= mul_out;
end
if(rst)begin
    mul_valid_r <= 1'b0;
end
else begin
    mul_valid_r <= window_valid;
end
end

// level2: add mul data
wire signed [31: 0] add_out;
generate
for(i=0; i<`KERNEL_SIZE; i++)begin
    add_out = $signed(add_out) + $signed(mul_out_r[i]);
end
endgenerate

reg signed [31: 0] add_out_r;
reg add_valid_r;

always_ff @(posedge clk)begin
if(window_valid)begin
    add_out_r <= add_out;
end
if(rst | window_new)begin
    add_valid_r <= 1'b0;
end
else begin
    add_valid_r <= mul_valid_r;
end
end

// level3: conv
reg [`KERNEL_SIZE-1: 0] conv_valid_r;
reg [`KERNEL_SIZE-1: 0][31: 0] conv_data_r;
wire [`KERNEL_SIZE-2: 0][31: 0] conv_data_n;

always_ff @(posedge clk)begin
    if(window_valid)begin
        conv_data_r[0] <= add_out_r;
    end

    if(rst | window_new)begin
        conv_valid_r <= 0;
    end
    else begin
        conv_valid_r[`KERNEL_SIZE-1] <= window_valid ? conv_valid_r[`KERNEL_SIZE-2] : 1'b0;
        if(window_valid)begin
            conv_valid_r[`KERNEL_SIZE-2: 0] <= {conv_valid_r[`KERNEL_SIZE-3: 0], add_valid_r};
        end
    end
end

generate
for(i=0; i<`KERNEL_SIZE-1; i=i+1)begin
    assign conv_data_n[i] = $signed(conv_data_r[i]) + $signed(add_out_r);
    always_ff @(posedge clk)begin
        if(window_valid)begin
            conv_data_r[i+1] <= conv_data_n[i];
        end
    end
end
endgenerate
    assign conv_valid = conv_valid_r[`KERNEL_SIZE-1];
    assign conv_data = conv_data_r[`KERNEL_SIZE-1];
endmodule
