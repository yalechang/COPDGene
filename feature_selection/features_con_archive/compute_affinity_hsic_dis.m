%% The motivation for using MATLAB is Python is too slow in computing the
%similarity matrix, it's better to use MATLAB instead.

%clear all, close all, clc;

% Load dataset
tp1 = load('data_dis.mat');
tp2 = load('features_name.mat');
tp3 = size(tp1.data);
data = cell2mat(tp1.data(1:tp3(1),1:(tp3(2)-3)));
gold = tp1.data(1:tp3(1),tp3(2));
features_name = tp2.features_name(1:(tp3(2)-1));
features_name_gold = tp2.features_name(tp3(2));

% Extract number of samples and number of features
[n_instances,n_features] = size(data);

% Use number to encode categorical features
tmp = unique(tp1.data(:,tp3(2)-2));
size_tmp = size(tmp);
tmp_vec = zeros(n_instances,1);
for i = 1:n_instances
    for j = 1:size_tmp(1)
        if strcmp(tp1.data(i,tp3(2)-2),tmp(j,1)) == 1
            tmp_vec(i,1) = j;
        end
    end
end

data = [data,tmp_vec,cell2mat(tp1.data(1:tp3(1),tp3(2)-1))];
[n_instances,n_features] = size(data);

%Compute HSIC matrix
mtr_hsic = ones(n_features);
arr_h = eye(n_instances)-1./n_instances;

for i = 1:n_features
    tmp = repmat(data(:,i),1,n_instances);
    arr_l = (tmp==tmp');
    arr_hlh = arr_h*arr_l*arr_h;
    for j = i:n_features
        tic
        tmp = repmat(data(:,j),1,n_instances);
        arr_k = (tmp==tmp');
        mtr_hsic(j,i) = (1./(n_instances-1)^2)*trace(arr_k*arr_hlh);
        mtr_hsic(i,j) = mtr_hsic(j,i);
        [i,j]
        toc
    end
end

%Compute Normalized HSIC matrix
mtr_nhsic = zeros(n_features);
for i = 1:n_features
    for j = i:n_features
        mtr_nhsic(i,j) = mtr_hsic(i,j)/sqrt(mtr_hsic(i,i)*mtr_hsic(j,j));
        mtr_nhsic(j,i) = mtr_nhsic(i,j);
    end
end
