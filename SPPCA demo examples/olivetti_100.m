clear all

%% preparing data
load('olivetti_faces.mat')
p= 100;
faces =imresize(faces,[p,p]);
x=reshape(faces,[p^2,400]);
cat = imread('lucky_cat3.jpg');
cat = double(imresize(cat,[p,p]));
NumCats = 20;
catStk = zeros(p^2,NumCats); % randomly rotated and vectorized cats 
                              % stacked into a long matrix
for i=1:NumCats
    angle= unifrnd(0,360,1);
    tmp = imrotate(cat,angle);
    tmp = imresize(tmp,[p,p]);
    catStk(:,i)=tmp(:);
end
x_contam =[x,catStk];

%rnd_ind = unidrnd(400,[16,1]);
rnd_ind =[
    111
    19
    39
   330
   278
   127
   381
    14
   176
   153
   307
   319
    75
   196
   179
   259];
figure
sgtitle('original faces'); 
for i=1:16
subplot(4,4,i)
imshow(faces(:,:,rnd_ind(i))/255)
end

figure
sgtitle('outliers'); 
for i=1:min(NumCats,16)
subplot(4,4,i)
tmp = catStk(:,i);
imshow(reshape(tmp,[p,p])/255)
end

%% robust PCA by Candes et al.
%{
lambda =0.008;
tol =10^(-6); max_iter =1000;
tic
[L_robPCA,S] = RobustPCA(x_contam,lambda,10*lambda, tol, max_iter);
time_robPCA =toc
Gamma_robPCA= orth(L_robPCA);

figure
sgtitle('low-rank reconstructions by robust PCA')
for i=1:16
subplot(4,4,i)
tmp = L_robPCA(:,rnd_ind(i));
imshow(reshape(tmp,[p,p])/255)
end

figure
sgtitle('robust PCA basis plots')
for i=1:16
    subplot(4,4,i)
    tmp=Gamma_robPCA(:,i);
    tmp=(tmp-min(tmp))/(max(tmp)-min(tmp));
    imshow(reshape(tmp,[p,p]))
end
%}

rank_robPCA =126;
%% PCA using uncontaminated data
tic
mu0 = mean(x,2);
[Gamma0,~] = svds(x- repmat(mu0,[1,400]),rank_robPCA-1,"largest");
time_PCA = toc
Gamma0=orth([mu0/norm(mu0), Gamma0]);
L0 = Gamma0*(Gamma0'*x);
figure
sgtitle('low-rank reconstructions by uncontaminated PCA')
for i=1:16
subplot(4,4,i)
tmp = L0(:,rnd_ind(i));
imshow(reshape(tmp,[p,p])/255)
end

% plotting basis
figure
sgtitle('uncontaminated PCA basis plots')
for i=1:16
    subplot(4,4,i)
    tmp=Gamma0(:,i);
    tmp=(tmp-min(tmp))/(max(tmp)-min(tmp));
    imshow(reshape(tmp,[p,p]))
end  

%% our SPPCA 
%{\
tic
scale_a = 0.7;
[mu_sppca, Sigma_sppca,wt_sppca, AR] = shape_V(x_contam', scale_a*p^2); % p^2 is the image dimensionality
[Gamma_sppca,~]=svds(Sigma_sppca,rank_robPCA-1,"largest");
time_sppca =toc
L_sppca = Gamma_sppca* (Gamma_sppca'*(x_contam-repmat(mu_sppca,[1,size(x_contam,2)])))...
    +repmat(mu_sppca,[1,size(x_contam,2)]);
Gamma_sppca = orth([mu_sppca,Gamma_sppca]);

figure
sgtitle('low-rank reconstructions by SPPCA')
for i=1:16
subplot(4,4,i)
tmp = L_sppca(:,rnd_ind(i));
imshow(reshape(tmp,[p,p])/255)
end

% plotting basis
figure
sgtitle('SPPCA basis plots')
for i=1:16
    subplot(4,4,i)
    tmp=Gamma_sppca(:,i);
    tmp=(tmp-min(tmp))/(max(tmp)-min(tmp));
    imshow(reshape(tmp,[p,p]))
end
   
%}

%% eigensubspace similarity comparison 
%sim_robPCA =svd(Gamma_robPCA'*Gamma0);
sim_SPPCA=svd(Gamma_sppca'*Gamma0);
mean_similarity = mean(sim_SPPCA)
%similarity = [mean(sim_robPCA), mean(sim_semiPCA)]




   


