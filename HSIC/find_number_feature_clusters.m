% Determine the number of feature clusters

% Compute Normalized HSIC matrix
mtr_nhsic = zeros(n_features);
for i = 1:n_features
    for j = i:n_features
        mtr_nhsic(i,j) = mtr_hsic(i,j)/sqrt(mtr_hsic(i,i)*mtr_hsic(j,j));
        mtr_nhsic(j,i) = mtr_nhsic(i,j);
    end
end

% Degree matrix
mtr_d = zeros(n_features);
for i = 1:n_features
    mtr_d(i,i) = sum(mtr_nhsic(i,:));
end

% Symmetric Laplacian
mtr_l = zeros(n_features);
for i = 1:n_features
    for j = 1:n_features
        mtr_l(i,j) = mtr_nhsic(i,j)/sqrt(mtr_d(i,i)*mtr_d(j,j));
    end
end

[eig_vec,eig_val] = eig(mtr_l);
tmp = zeros(1,n_features);
for i = 1:n_features
    tmp(i) = eig_val(n_features+1-i,n_features+1-i);
end
figure;
stem(tmp);
grid on;
xlabel('Index of Features');
ylabel('Eigenvalues');
title('Eigenvalues of Laplacian Matrix');

% Plot difference
tmp_diff = zeros(1,n_features);
for i = 2:n_features
    tmp_diff(i) = abs(tmp(i)-tmp(i-1));
end
figure;
stem(tmp_diff);
grid on;
xlabel('Index of Features');
ylabel('Eigenvale Gaps');
title('Eigenvalue Gaps of Laplacian Matrix');
