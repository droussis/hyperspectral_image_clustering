clc;
format compact;
clear;
close all;
%rand('seed',1);
rng('default');
rng(2);


%% Loading and Preprocessing %%

load Salinas_Data.mat;
[p1,n1,l]=size(Salinas_Image);
X=reshape(Salinas_Image,p1*n1,l);
X=X';
[p2,n2]=size(Salinas_Labels);
y=reshape(Salinas_Labels,p2*n2,1);
y=y';


%% Remove zero class labels %%

zero_idx=find(~y);
nonzero_idx=find(y);
X(:,zero_idx)=[];
y(:,zero_idx)=[];
[l,N]=size(X);


%% Visualizations %%

im=Salinas_Labels;
figure(1), imagesc(im);
title('Salinas Labels');
hold off

vol=Salinas_Image;
figure(2),volshow(vol,'Colormap',colormap);
hold off


%% NOTE: Use only one of the two preprocessing methods below %%

%% Filtering and Normalization %%

X=imgaussfilt(X,3.5,'FilterSize',9);
X=normalize(X);


%% Dimensionality Reduction with PCA %%

[eigenval,eigenvec,explain,Y,mean_vec]=pca_fun(X,4);
X=Y;
[l,N]=size(X);


%% k-Means Algorithm %%

rand_k_means_all=[]
jaccard_k_means_all=[]

for m=3:10
    
    costs=[];
    bels=[];

    for i=1:10
        theta_init=rand(l,m);
        [theta,bel,J]=k_means(X,theta_init);
        bels=[bels;bel];
        costs=[costs;J];
    end

    figure(3),plot(costs)
    [cost_m,bel_index]=min(costs)
    bel=bels(bel_index,:);

    cl_label=bel';
    cl_label_tot=zeros(p1*n1,1);
    cl_label_tot(nonzero_idx)=cl_label;
    im_cl_label=reshape(cl_label_tot,p1,n1);
    figure(4), imagesc(im_cl_label); 

    [rand_k_means,jaccard_k_means]=metrics(bel,y)
    
    rand_k_means_all=[rand_k_means_all;
                      rand_k_means]
    jaccard_k_means_all=[jaccard_k_means_all;
                         jaccard_k_means]

end


%% Possibilistic c-Means Algorithm %%

rand_possibilistic_all=[]
jaccard_possibilistic_all=[]

for m=3:10
    
    costs=[];
    bels=[];

    for i=1:10
        theta_init=rand(l,m);
        theta_init=k_means(X,theta_init);
        theta_init=fuzzy_c_means(X,theta_init,2);
        [theta,bel,J]=possibilistic_c_means(X,theta_init);
        bels=[bels;bel];
        costs=[costs;J];
    end

    figure(5),plot(costs)
    [cost_m,bel_index]=min(costs)
    bel=bels(bel_index,:);

    cl_label=bel';
    cl_label_tot=zeros(p1*n1,1);
    cl_label_tot(nonzero_idx)=cl_label;
    im_cl_label=reshape(cl_label_tot,p1,n1);
    figure(6), imagesc(im_cl_label); 

    [rand_possibilistic,jaccard_possibilistic]=metrics(bel,y)
    
    rand_possibilistic_all=[rand_possibilistic_all;
                            rand_possibilistic]
    jaccard_possibilistic_all=[jaccard_possibilistic_all;
                               jaccard_possibilistic]

end


%% Fuzzy c-Means Algorithm %%

rand_fuzzy_all=[]
jaccard_fuzzy_all=[]

for m=3:10
    
    costs=[];
    bels=[];

    for i=1:10
        theta_init=rand(l,m);
        [theta,bel,J]=fuzzy_c_means(X,theta_init,2);
        bels=[bels;bel];
        costs=[costs;J];
    end

    figure(7),plot(costs)
    [cost_m,bel_index]=min(costs)
    bel=bels(bel_index,:);

    cl_label=bel';
    cl_label_tot=zeros(p1*n1,1);
    cl_label_tot(nonzero_idx)=cl_label;
    im_cl_label=reshape(cl_label_tot,p1,n1);
    figure(8), imagesc(im_cl_label); 

    [rand_fuzzy,jaccard_fuzzy]=metrics(bel,y)
    
    rand_fuzzy_all=[rand_fuzzy_all;
                    rand_fuzzy]
    jaccard_fuzzy_all=[jaccard_fuzzy_all;
                       jaccard_fuzzy]

end


%% Probabilistic c-Means Algorithm %%

rand_probabilistic_all=[];
jaccard_probabilistic_all=[];

for m=3:10

    options=statset('Display','final','MaxIter',500,'TolFun',5e-7);
    gmfit=fitgmdist(X',m,'Start','plus','CovarianceType', 'full', ...
                    'Options',options,'RegularizationValue',1e-4, ...
                    'Replicates',10);
    bel=cluster(gmfit,X');
    
    [rand_probabilistic,jaccard_probabilistic]=metrics(bel',y)
    
    rand_probabilistic_all=[rand_probabilistic_all;
                            rand_probabilistic]
    jaccard_probabilistic_all=[jaccard_probabilistic_all;
                               jaccard_probabilistic];
    
    cl_label=bel;
    cl_label_tot=zeros(p1*n1,1);
    cl_label_tot(nonzero_idx)=cl_label;
    im_cl_label=reshape(cl_label_tot,p1,n1);
    figure(9), imagesc(im_cl_label);                        
                           
end


%% Visualization of the 1st component of PCA %%

[eigenval,eigenvec,explain,Y,mean_vec]=pca_fun(X,1);
Y_tot=zeros(p1*n1,1);
Y_tot(nonzero_idx)=Y(1,:);
Y_reshaped=reshape(Y_tot(:,1),p1,n1);
figure(10), imagesc(Y_reshaped);


%% Ensemble Fuzzy c-Means -> k-Means %%

rand_fuzzy_k_means_all=[]

for m=3:10
    
    costs=[];
    bels=[];
    for i=1:10
        theta_init=rand(l,m);
        [theta,bel,J]=fuzzy_c_means(X,theta_init,2);
        [theta,bel,J]=k_means(X,theta);
        bels=[bels;bel];
        costs=[costs;J];
    end

    figure(11),plot(costs)
    [cost_m,bel_index]=min(costs)
    bel=bels(bel_index,:);
    
    [rand_fuzzy_k_means]=metrics(bel,y)
    
    rand_fuzzy_k_means_all=[rand_fuzzy_k_means_all;
                            rand_fuzzy_k_means]
    
    cl_label=bel';
    cl_label_tot=zeros(p1*n1,1);
    cl_label_tot(nonzero_idx)=cl_label;
    im_cl_label=reshape(cl_label_tot,p1,n1);
    figure(12), imagesc(im_cl_label); 
                        
end
