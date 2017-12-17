function W = randInitializeWeights(L_in, L_out)
## Random initializing weight matrices with a dimension of 
# (From Layer + 1) * (To Layer) 
W = zeros(1 + L_in,L_out);
epsilon_init = 0.12;
W = rand(1 + L_in,L_out) * 2 * epsilon_init - epsilon_init;