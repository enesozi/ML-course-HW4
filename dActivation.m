function g = dActivation(z)
%activation Compute derivative value of tanh function
g = zeros(size(z));
g = 1- tanh(z).^2;

end