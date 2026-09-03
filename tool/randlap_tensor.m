function X = randlap_tensor(dim,beta)
%Probability distribution function of Laplace noise: F(x) = 1/(2*beta)*e^(-|x|/beta)
leng       = length(dim);
n=1;
for i = 1:leng
 n=dim(i)*n;
end
x=randlap(n,beta);
X=reshape(x,dim);
end




