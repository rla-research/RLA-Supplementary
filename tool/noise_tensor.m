function [X] = noise_tensor(dim,L_beta,G_std)
X=randlap_tensor(dim,L_beta)+G_std*randn(dim);
end

