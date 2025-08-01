`timescale 1ns / 1ns

module adaptive_leaky_integrate_and_fire_async (
    input wire clk,
    input wire rst,
    input real i_in,
    output reg spike
);

// Parameters
parameter real tau = 5e-9;   // Membrane time constant (s)
parameter real Vrest = 0;     // Resting membrane potential (mV)
parameter real Vreset = 0;    // Reset potential after spike (mV)
parameter real R = 10.0;      // Membrane resistance (Mohm)
parameter real dt = 1e-10;     // Time step (s)
parameter real leak = 0;   // Leak constant (evaluated at each 0.1 ns)
parameter real alpha = 0.01;  // Adaptation rate
parameter real tau_th = 1.0;  // Adaptation time constant

// Internal signals
real V = Vrest;       // Membrane potential
real Vth = 5e-5;     // Initial threshold
real dV;
real dVth;

// Asynchronous signal integration
always @(i_in) begin
    dV = (R * i_in - (V - Vrest) - R * leak * (V - Vrest)) / tau;
    V = V + dV * dt;

    // Threshold adaptation update
    dVth = (alpha * spike - Vth) / tau_th;
    Vth = Vth + dVth * dt;
end

// Spike logic
always @(posedge clk) begin
    if (rst) begin
        V <= Vrest;
        spike <= 0;
    end else begin
        if (V >= Vth) begin
            // $display("Time: %0t, Dynamic", $time);
            spike <= 1;
            V <= Vreset;
        end else begin
            spike <= 0;
            // $display("Time: %0t, Static", $time);
        end
    end
end

endmodule
