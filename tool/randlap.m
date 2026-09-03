function x = randlap(n,beta) 
%Probability distribution function of Laplace noise: F(x) = 1/(2*beta)*e^(-|x-mu|/beta)
mu = 0; % mean=0  var=2*beta^2;
% Generate uniformly distributed random numbers from [-0.5, 0.5]
z = rand(n, 1)-0.5;
% Apply probability integral transformation
x = mu - beta .* sign(z) .* log(1 - 2*abs(z));
end 


% function x = randlap(n,beta)
% 
% z = rand(n,1);
% x = zeros(n,1);
% 
% in = z<=.5;
% x(in) =  beta *log(2*z(in));
% ip = z> .5;
% x(ip) = -beta *log(2*(1-z(ip)));



