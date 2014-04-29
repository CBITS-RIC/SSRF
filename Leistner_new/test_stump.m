clear all;
close all;

load '../Data/G50C/g50c.mat';
Y = y;

ind_train = [1:round(length(X)/3), round(length(X)/2):round(5*length(X)/6)];

% Xtrain = X(ind_train, :);
% Ytrain = Y(ind_train);
% 
% Xtest = X;
% Ytest = Y;
% Xtest(ind_train, :) = [];
% Ytest(ind_train) = [];

Xtrain = X(idxLabs(1,:),:);
Ytrain = Y(idxLabs(1,:));
Xtest = X(idxUnls(1,:),:);
Ytest = Y(idxUnls(1,:));

% return;

for i=1:500,
    stump{i} = ClassificationTree.fit(Xtrain, Ytrain, 'Prune','off', 'nvartosamp', 4, 'MinParent', size(Xtrain,1));
    [ys(:,i) p(:,:,i)] = predict(stump{i}, Xtest);
%     sum(ys==Ytest)/length(Ytest)
%     classes = stump{i}.NodeClass;
end
yf = mode(ys,2);
sum(yf==Ytest)/length(Ytest)

