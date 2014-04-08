clear all; close all
%1st distribution
N = 1000;
mu = [1 2];
Sigma = [2 .5; .5 1];
z1 = repmat(mu,N,1) + randn(N,2)*Sigma;

figure
plot(z1(:,1),z1(:,2),'or')


%2nd distribution
N = 1000;
mu = [16 -6];
Sigma = [1 0; 0 1];
z2 = repmat(mu,N,1) + randn(N,2)*Sigma;

hold on
plot(z2(:,1),z2(:,2),'ob')

labels = [1*ones(size(z1,1),1); 2*ones(size(z2,1),1)];
data = [z1; z2];

%reshuffle data
ind = randperm(size(data,1));
data = data(ind,:); labels = labels(ind);
save 'data_art.mat' data;
save 'labels_art.mat' labels;