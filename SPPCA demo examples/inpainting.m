%% example was taken and modified from RobustPCA package 

clear all

% read image and add the mask
groundtruth = double(imread('moon.tif'))/255;
groundtruth = groundtruth(141:140+256, 51:50+256);
msk = zeros(size(groundtruth));
msk(65:192,65:192) = imresize(imread('text.png'),0.5);
img_corrupted = groundtruth;
img_corrupted(msk > 0) = 0;

% create a matrix X from overlapping patches
ws = 16; % window size
no_patches = size(groundtruth, 1) / ws;
k = 1;
for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2; % size: 1x16 % ws/2 is the stride
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2; % size: 1x16
        patch = img_corrupted(r1, r2);
        X(k,:) = patch(:);
        k = k + 1;
    end
end

% unmask the following lines to apply Robust PCA
%{
lambda = 0.02; 
[L, S] = RobustPCA(X, lambda, 1.0, 1e-5);

% reconstruct the image from the overlapping patches in matrix L
img_reconstructed = zeros(size(groundtruth));
img_noise = zeros(size(groundtruth));
k = 1;
for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        % average patches to get the image back from L and S
        % todo: in the borders less than 4 patches are averaged
        patch = reshape(L(k,:), ws, ws);
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        img_reconstructed(r1, r2) = img_reconstructed(r1, r2) + 0.25*patch;
        patch = reshape(S(k,:), ws, ws);
        img_noise(r1, r2) = img_noise(r1, r2) + 0.25*patch;
        k = k + 1;
    end
end
%}

%% SPPCA for low-rank reconstruction
scale_a = 1; 
max_iter =200;
[mu, Sigma,wt, AR] = shape_V_beta(X,scale_a*size(X,2), max_iter); % scale is of order bigOh(dimensionality)
wt =wt/sum(wt); 
k=28; % k = rank(L) from robust PCA
[Gamma,eigVal]=svds(Sigma,k-1,"largest");
L2 = (diag(wt*size(X,1))*(X-repmat(mu',[size(X,1),1]))*Gamma) *Gamma'+repmat(mu',[size(X,1),1]);
     % using weighted sample
img_reconstructed2 = zeros(size(groundtruth));
img_noise = zeros(size(groundtruth));
k = 1;
for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        % average patches to get the image back from L and S
        % todo: in the borders less than 4 patches are averaged
        patch = reshape(L2(k,:), ws, ws);
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        img_reconstructed2(r1, r2) = img_reconstructed2(r1, r2) + 0.25*patch;
        k = k + 1;
    end
end

figure;
subplot(2,2,1), imshow(img_corrupted), title('corrupted image')
subplot(2,2,2), imshow(groundtruth), title('ground truth')
subplot(2,2,3), imshow(img_reconstructed2), title('recovered low-rank by SPPCA')
%subplot(2,2,4), imshow(img_reconstructed), title('recovered low-rank by robust PCA')






