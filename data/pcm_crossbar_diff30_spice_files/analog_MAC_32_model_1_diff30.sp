*Fully-connected Classifier
.lib './models' ptm14hp

*Differential Amplifier with Gain=10
.SUBCKT diff3 in+ in- out vdd vss
R1 in- n1 1k
*R5 n10 n1 0.7
* C1 n1 0 50f IC=0V
R2 in+ n2 1k
* R6 n20 n2 0.7
* C2 n2 0 50f IC=0V

E1 out1 0 OPAMP n2 n1 1
$ Loop
R3 n1 out1 30k
R4 n2 0 30k
*G1 out 0 n2 n1 1
* Iout out 0 0.1m

* VCCS controlling current (slew-rate limiter)
* G1 out 0 nbuf n1 10n    

* Feedback network (adjust gain of VCCS to control charging speed)
* E2 nbuf 0 out1 0 1  

* R6 op_out out 500

* Ebuf nbuf 0 op_out 0 1


* X1 out2 nbuf vdd vdd nfet nfin=500
* X2 out2 nbuf vss vss pfet nfin=500
* *G1 out 0 nbuf 0 10u


* * X1 out op_out vdd vss nfet nfin=10
* Ivss out2 vss DC 10u
* Ivdd out2 vdd DC 10u

* E3 out3 0 out2 0 1.074
* V5 vss4 0 80.82m
* E4 out 0 out3 vss4 1






* RC Delay 
R7 out1 out 50
C3 out 0 1p IC=0V

C4 out 0 1f IC=0V



.ENDS diff3

* Neuron design from IMAC-Sim
.SUBCKT neuron in out vsup+ vsup-

X1 out input vsup- vsup- nfet nfin=10
X2 out input vsup+ vsup+ pfet nfin=10
Rlow in2 input 5000.0
vin in in2 -0.4
Rhigh input out 15000.0

.ENDS neuron
* Layer definition for neural network
.SUBCKT layer vdd vss 0 in1 in2 in3 in4 in5 in6 in7 in8 in9 in10 in11 in12 in13 in14 in15 in16 in17 in18 in19 in20 in21 in22 in23 in24 in25 in26 in27 in28 in29 in30 in31 in32 out1
**********Positive Weighted Array**********
Rwpos1_1 in1_1 sp1_1 5000.0
Rwpos2_1 in2_1 sp2_1 5000.0
Rwpos3_1 in3_1 sp3_1 5000.0
Rwpos4_1 in4_1 sp4_1 5000.0
Rwpos5_1 in5_1 sp5_1 5000.0
Rwpos6_1 in6_1 sp6_1 5000.0
Rwpos7_1 in7_1 sp7_1 5000.0
Rwpos8_1 in8_1 sp8_1 5000.0
Rwpos9_1 in9_1 sp9_1 5000.0
Rwpos10_1 in10_1 sp10_1 5000.0
Rwpos11_1 in11_1 sp11_1 5000.0
Rwpos12_1 in12_1 sp12_1 5000.0
Rwpos13_1 in13_1 sp13_1 5000.0
Rwpos14_1 in14_1 sp14_1 5000.0
Rwpos15_1 in15_1 sp15_1 5000.0
Rwpos16_1 in16_1 sp16_1 5000.0
Rwpos17_1 in17_1 sp17_1 5000.0
Rwpos18_1 in18_1 sp18_1 5000.0
Rwpos19_1 in19_1 sp19_1 5000.0
Rwpos20_1 in20_1 sp20_1 5000.0
Rwpos21_1 in21_1 sp21_1 5000.0
Rwpos22_1 in22_1 sp22_1 5000.0
Rwpos23_1 in23_1 sp23_1 5000.0
Rwpos24_1 in24_1 sp24_1 5000.0
Rwpos25_1 in25_1 sp25_1 5000.0
Rwpos26_1 in26_1 sp26_1 5000.0
Rwpos27_1 in27_1 sp27_1 5000.0
Rwpos28_1 in28_1 sp28_1 5000.0
Rwpos29_1 in29_1 sp29_1 5000.0
Rwpos30_1 in30_1 sp30_1 5000.0
Rwpos31_1 in31_1 sp31_1 5000.0
Rwpos32_1 in32_1 sp32_1 5000.0
**********Negative Weighted Array**********
Rwneg1_1 in1_1 sn1_1 5000.0
Rwneg2_1 in2_1 sn2_1 5000.0
Rwneg3_1 in3_1 sn3_1 5000.0
Rwneg4_1 in4_1 sn4_1 5000.0
Rwneg5_1 in5_1 sn5_1 5000.0
Rwneg6_1 in6_1 sn6_1 5000.0
Rwneg7_1 in7_1 sn7_1 5000.0
Rwneg8_1 in8_1 sn8_1 5000.0
Rwneg9_1 in9_1 sn9_1 5000.0
Rwneg10_1 in10_1 sn10_1 5000.0
Rwneg11_1 in11_1 sn11_1 5000.0
Rwneg12_1 in12_1 sn12_1 5000.0
Rwneg13_1 in13_1 sn13_1 5000.0
Rwneg14_1 in14_1 sn14_1 5000.0
Rwneg15_1 in15_1 sn15_1 5000.0
Rwneg16_1 in16_1 sn16_1 5000.0
Rwneg17_1 in17_1 sn17_1 5000.0
Rwneg18_1 in18_1 sn18_1 5000.0
Rwneg19_1 in19_1 sn19_1 5000.0
Rwneg20_1 in20_1 sn20_1 5000.0
Rwneg21_1 in21_1 sn21_1 5000.0
Rwneg22_1 in22_1 sn22_1 5000.0
Rwneg23_1 in23_1 sn23_1 5000.0
Rwneg24_1 in24_1 sn24_1 5000.0
Rwneg25_1 in25_1 sn25_1 5000.0
Rwneg26_1 in26_1 sn26_1 5000.0
Rwneg27_1 in27_1 sn27_1 5000.0
Rwneg28_1 in28_1 sn28_1 5000.0
Rwneg29_1 in29_1 sn29_1 5000.0
Rwneg30_1 in30_1 sn30_1 5000.0
Rwneg31_1 in31_1 sn31_1 5000.0
Rwneg32_1 in32_1 sn32_1 5000.0
**********Positive Biases**********
Rbpos1 vd1 sp32_1 5000.0
**********Negative Biases**********
Rbneg1 vd1 sn32_1 5000.0
**********Parasitic Resistances for Vertical Lines**********
Rin1_1 in1 in1_1 9.241569
Rin2_1 in2 in2_1 9.241569
Rin3_1 in3 in3_1 9.241569
Rin4_1 in4 in4_1 9.241569
Rin5_1 in5 in5_1 9.241569
Rin6_1 in6 in6_1 9.241569
Rin7_1 in7 in7_1 9.241569
Rin8_1 in8 in8_1 9.241569
Rin9_1 in9 in9_1 9.241569
Rin10_1 in10 in10_1 9.241569
Rin11_1 in11 in11_1 9.241569
Rin12_1 in12 in12_1 9.241569
Rin13_1 in13 in13_1 9.241569
Rin14_1 in14 in14_1 9.241569
Rin15_1 in15 in15_1 9.241569
Rin16_1 in16 in16_1 9.241569
Rin17_1 in17 in17_1 9.241569
Rin18_1 in18 in18_1 9.241569
Rin19_1 in19 in19_1 9.241569
Rin20_1 in20 in20_1 9.241569
Rin21_1 in21 in21_1 9.241569
Rin22_1 in22 in22_1 9.241569
Rin23_1 in23 in23_1 9.241569
Rin24_1 in24 in24_1 9.241569
Rin25_1 in25 in25_1 9.241569
Rin26_1 in26 in26_1 9.241569
Rin27_1 in27 in27_1 9.241569
Rin28_1 in28 in28_1 9.241569
Rin29_1 in29 in29_1 9.241569
Rin30_1 in30 in30_1 9.241569
Rin31_1 in31 in31_1 9.241569
Rin32_1 in32 in32_1 9.241569
Rbias1 vdd vd1 9.241569
**********Parasitic Resistances for I+ and I- Lines**********
Rsp1_1 sp1_1 sp2_1 11.551961
Rsn1_1 sn1_1 sn2_1 11.551961
Rsp2_1 sp2_1 sp3_1 11.551961
Rsn2_1 sn2_1 sn3_1 11.551961
Rsp3_1 sp3_1 sp4_1 11.551961
Rsn3_1 sn3_1 sn4_1 11.551961
Rsp4_1 sp4_1 sp5_1 11.551961
Rsn4_1 sn4_1 sn5_1 11.551961
Rsp5_1 sp5_1 sp6_1 11.551961
Rsn5_1 sn5_1 sn6_1 11.551961
Rsp6_1 sp6_1 sp7_1 11.551961
Rsn6_1 sn6_1 sn7_1 11.551961
Rsp7_1 sp7_1 sp8_1 11.551961
Rsn7_1 sn7_1 sn8_1 11.551961
Rsp8_1 sp8_1 sp9_1 11.551961
Rsn8_1 sn8_1 sn9_1 11.551961
Rsp9_1 sp9_1 sp10_1 11.551961
Rsn9_1 sn9_1 sn10_1 11.551961
Rsp10_1 sp10_1 sp11_1 11.551961
Rsn10_1 sn10_1 sn11_1 11.551961
Rsp11_1 sp11_1 sp12_1 11.551961
Rsn11_1 sn11_1 sn12_1 11.551961
Rsp12_1 sp12_1 sp13_1 11.551961
Rsn12_1 sn12_1 sn13_1 11.551961
Rsp13_1 sp13_1 sp14_1 11.551961
Rsn13_1 sn13_1 sn14_1 11.551961
Rsp14_1 sp14_1 sp15_1 11.551961
Rsn14_1 sn14_1 sn15_1 11.551961
Rsp15_1 sp15_1 sp16_1 11.551961
Rsn15_1 sn15_1 sn16_1 11.551961
Rsp16_1 sp16_1 sp17_1 11.551961
Rsn16_1 sn16_1 sn17_1 11.551961
Rsp17_1 sp17_1 sp18_1 11.551961
Rsn17_1 sn17_1 sn18_1 11.551961
Rsp18_1 sp18_1 sp19_1 11.551961
Rsn18_1 sn18_1 sn19_1 11.551961
Rsp19_1 sp19_1 sp20_1 11.551961
Rsn19_1 sn19_1 sn20_1 11.551961
Rsp20_1 sp20_1 sp21_1 11.551961
Rsn20_1 sn20_1 sn21_1 11.551961
Rsp21_1 sp21_1 sp22_1 11.551961
Rsn21_1 sn21_1 sn22_1 11.551961
Rsp22_1 sp22_1 sp23_1 11.551961
Rsn22_1 sn22_1 sn23_1 11.551961
Rsp23_1 sp23_1 sp24_1 11.551961
Rsn23_1 sn23_1 sn24_1 11.551961
Rsp24_1 sp24_1 sp25_1 11.551961
Rsn24_1 sn24_1 sn25_1 11.551961
Rsp25_1 sp25_1 sp26_1 11.551961
Rsn25_1 sn25_1 sn26_1 11.551961
Rsp26_1 sp26_1 sp27_1 11.551961
Rsn26_1 sn26_1 sn27_1 11.551961
Rsp27_1 sp27_1 sp28_1 11.551961
Rsn27_1 sn27_1 sn28_1 11.551961
Rsp28_1 sp28_1 sp29_1 11.551961
Rsn28_1 sn28_1 sn29_1 11.551961
Rsp29_1 sp29_1 sp30_1 11.551961
Rsn29_1 sn29_1 sn30_1 11.551961
Rsp30_1 sp30_1 sp31_1 11.551961
Rsn30_1 sn30_1 sn31_1 11.551961
Rsp31_1 sp31_1 sp32_1 11.551961
Rsn31_1 sn31_1 sn32_1 11.551961
Rsp32_1 sp32_1 sp1_p1 11.551961
Rsn32_1 sn32_1 sn1_p1 11.551961
**********Weight Differntial Op-AMPS and Connecting Resistors****************
XDIFFw1_p1 sp1_p1 sn1_p1 nin1_1 vdd vss diff3
Rconn1_p1 nin1_1 out1 1m
.ENDS layer
