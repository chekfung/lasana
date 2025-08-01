*Differential Amplifier with Gain=10
.SUBCKT diff3 in+ in- out
R1 in- n1 1k
R2 in+ n2 1k
E1 out1 0 OPAMP n2 n1

R3 n1 out1 10k
R4 n2 0 10k

* RC Delay 
R7 out1 out 50
C3 out 0 1p IC=0V
.ENDS diff3

