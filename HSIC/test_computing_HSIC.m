clear all, close all, clc;

load data_train_continuous_scaled.mat data_train_continuous_scaled
data = data_train_continuous_scaled;
[n_instances,n_features] = size(data);
sigma_hsic = 1.0;


% Compute HSIC matrix
mtr_hsic = zeros(n_features);
arr_h = eye(n_instances)-1/n_instances;
for i = 1:n_features
    tmp = data(:,i);
    arr_l = exp(-1/(2*sigma_hsic^2)*(repmat(tmp,1,n_instances)-repmat(tmp',n_instances,1)));
    arr_hlh = arr_h*arr_l*arr_h;
    for j = i:n_features
        tic
        tp = data(:,j);
        arr_k = exp(-1/(2*sigma_hsic^2)*(repmat(tp,1,n_instances)-repmat(tp',n_instances,1)));
        mtr_hsic(j,i) = 1/(n_instances-1)^2*trace(arr_k*arr_hlh);
        mtr_hsic(i,j) = mtr_hsic(j,i);
        toc
        [i,j]
    end
end

% Save the file
save mtr_hsic.mat mtr_hsic